import os
import re
import sys
import h5py
import pandas as pd
import numpy as np
import math

class PostProcess:
    @staticmethod
    def read_DAT(file):
        """
        Lê arquivos .dat de pós-processamento do OpenFOAM como force.dat,
        moment.dat, residuals.dat, etc.
        """
        try:
            with open(file, 'r', encoding='utf-8') as filename:
                logFile = filename.readlines()
        except Exception as e:
            print(f"Erro ao ler o arquivo {file}: {e}")
            return pd.DataFrame()

        if not logFile:
            return pd.DataFrame()

        data = []
        header_line = []
        header_found = False
        for line in logFile:
            # Procura pela linha de cabeçalho que começa com '#' e contém 'Time'
            if line.strip().startswith('#'):
                if 'Time' in line:
                    header_line = re.sub(r'[()#]', '', line).split()
                    header_found = True
                continue # Pula para a próxima linha após encontrar um comentário

            if not header_found:
                continue

            # Processa as linhas de dados
            l = list(filter(None, re.sub(r'[()#]', '', line).split()))
            if l:
                data.append(l)

        if not data or not header_line:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=header_line)
        # Converte para numérico, tratando erros, e define 'Time' como índice
        df = df.apply(pd.to_numeric, errors='coerce')
        df.set_index('Time', inplace=True)
        return df

    @staticmethod
    def read_zone_dat(file):
        """
        Lê arquivos .dat de zonas (como os de fieldMinMax), com suporte para campos vetoriais.
        Expande cabeçalhos como 'max(U)' em 'max(U)_x', 'max(U)_y', 'max(U)_z'.
        """
        with open(file, 'r') as f:
            lines = f.readlines()

        if not lines:
            return None

        header_line_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('#') and 'Time' in line:
                header_line_index = i
                break
        
        # Se não encontrar um cabeçalho de zona, tenta usar o leitor padrão
        if header_line_index == -1: return PostProcess.read_DAT(file)

        header_fields = lines[header_line_index].strip().replace('#', '').strip().split()
        
        data_rows = []
        for line in lines[header_line_index + 1:]:
            if line.strip() and not line.strip().startswith('#'):
                cleaned_line = re.sub(r'[()]', '', line)
                data_rows.append(cleaned_line.strip().split())
        
        if not data_rows: return None

        num_data_cols = len(data_rows[0])
        
        expanded_header = []
        col_cursor = 0
        field_cursor = 0
        while field_cursor < len(header_fields):
            field_name = header_fields[field_cursor]
            
            if field_name == 'Time':
                expanded_header.append('Time')
                col_cursor += 1
                field_cursor += 1
                continue
            
            is_vector = '(U)' in field_name and (col_cursor + 2 < num_data_cols)

            if is_vector:
                expanded_header.extend([f"{field_name}_x", f"{field_name}_y", f"{field_name}_z"])
                col_cursor += 3
            else:
                expanded_header.append(field_name)
                col_cursor += 1
            field_cursor += 1
        
        if len(expanded_header) != num_data_cols:
            return PostProcess.read_DAT(file)

        df = pd.DataFrame(data_rows, columns=expanded_header)
        df = df.apply(pd.to_numeric, errors='coerce').set_index('Time')
        
        return df

    @staticmethod
    def convert_dat_h5(fname_force, fname_moment, fname_h5):
        data_force = PostProcess.read_DAT(fname_force)
        data_moment = PostProcess.read_DAT(fname_moment)

        data_force = data_force.rename(columns={'total_x':'FX0', 'total_y':'FY0', 'total_z':'FZ0'})
        data_moment = data_moment.rename(columns={'total_x':'MX0', 'total_y':'MY0', 'total_z':'MZ0'})

        data_force.index = data_force.index.astype(float)
        data_moment.index = data_moment.index.astype(float)

        data_juncao = pd.merge(
            data_force[['FX0','FY0','FZ0']],
            data_moment[['MX0','MY0','MZ0']],
            right_index=True, left_index=True
        )

        compound_dtype = data_juncao.to_records()
        
        with h5py.File(fname_h5, 'w') as h5:
            h5.create_dataset('force_moment_data', data=compound_dtype)

    @staticmethod
    def read_areaAverage(file, fields=''):
        if not fields:
            raise Exception('Campos nao definidos em read_areaAverage')
        fields = re.sub(r'\s+', '', fields).lower()
        columns = []
        for char in fields:
            if char == 'u':
                columns.extend(['U_x', 'U_y', 'U_z'])
            elif char == 'p':
                columns.extend(['p'])

        data = PostProcess.read_DAT(file)
        data.columns = columns
        return data

    @staticmethod
    def process_case_to_h5_unified(base_path):
        """
        Processa um diretório para encontrar e unificar dados de todas as pastas 'case_*',
        ou processa uma única pasta 'case_*' se o caminho for fornecido diretamente.
        Unifica dados de 'force/moment' e 'zone' em arquivos HDF5.
        """
        case_paths = []
        # Verifica se o caminho fornecido já é uma pasta de caso
        if os.path.basename(os.path.normpath(base_path)).startswith('case_'):
            case_paths.append(base_path)
        else:
            # Se não for, procura por pastas 'case_*' dentro do diretório
            case_folders = [d for d in os.listdir(base_path) if d.startswith('case_') and os.path.isdir(os.path.join(base_path, d))]
            if not case_folders:
                print(f"Nenhuma pasta 'case_*' encontrada em '{base_path}'")
                return
            case_paths = [os.path.join(base_path, f) for f in sorted(case_folders)]

        for case_path in case_paths:
            print(f"\n--- Processando Case: {os.path.basename(case_path)} ---")
            case_name = os.path.basename(os.path.normpath(case_path))
            h5_filename = os.path.join(os.path.dirname(case_path), f'{case_name}_unified.h5')

            with h5py.File(h5_filename, 'w') as h5_file:
                sim_dirs_names = [d for d in os.listdir(case_path) if d.startswith('sim_') and os.path.isdir(os.path.join(case_path, d))]
                if not sim_dirs_names:
                    print(f"Nenhuma pasta 'sim_*' encontrada em {case_path}")
                    continue

                sim_dirs = sorted(sim_dirs_names, key=lambda x: int(x.split('_')[1]))
                max_sim_num = max([int(d.split('_')[1]) for d in sim_dirs])
                ndigits = math.floor(math.log10(max_sim_num)) + 1 if max_sim_num > 0 else 1

                for sim_dir_name in sim_dirs:
                    print(f'--- Lendo Sim: {sim_dir_name} ---')
                    sim_path = os.path.join(case_path, sim_dir_name)
                    post_processing_path = os.path.join(sim_path, 'postProcessing')

                    if not os.path.isdir(post_processing_path):
                        print(f"Pasta 'postProcessing' não encontrada em {sim_dir_name}, pulando.")
                        continue

                    all_data_merged = pd.DataFrame()

                    # 1. Processa dados genéricos (não-zona)
                    data_folders = [d for d in os.listdir(post_processing_path) if os.path.isdir(os.path.join(post_processing_path, d)) and not d.startswith('Zone_')]
                    for folder in data_folders:
                        search_dirs = [os.path.join(post_processing_path, folder)]
                        if os.path.isdir(os.path.join(search_dirs[0], '0')):
                            search_dirs.append(os.path.join(search_dirs[0], '0'))
                        
                        for s_dir in search_dirs:
                            for dat_file_name in [f for f in os.listdir(s_dir) if f.endswith('.dat')]:
                                df = PostProcess.read_DAT(os.path.join(s_dir, dat_file_name))

                                if dat_file_name == 'solverInfo.dat':
                                    cols_to_keep = [c for c in df.columns if c.endswith('_final')]
                                    df = df[cols_to_keep]

                                if not df.empty: 
                                    prefix = f"{os.path.splitext(dat_file_name)[0]}_"
                                    df = df.add_prefix(prefix)
                                    all_data_merged = pd.merge(all_data_merged, df, left_index=True, right_index=True, how='outer')

                    # 2. Processa dados de Zonas
                    zone_pattern = re.compile(r'^Zone_(\d+)_.*')
                    zone_folders = [d for d in os.listdir(post_processing_path) if zone_pattern.match(d) and os.path.isdir(os.path.join(post_processing_path, d))]
                    data_by_zone = {}
                    for zone_folder in zone_folders:
                        match = zone_pattern.match(zone_folder)
                        zone_id = int(match.group(1))
                        
                        search_dirs = [os.path.join(post_processing_path, zone_folder)]
                        if os.path.isdir(os.path.join(search_dirs[0], '0')):
                            search_dirs.append(os.path.join(search_dirs[0], '0'))

                        for s_dir in search_dirs:
                            for dat_file_name in [f for f in os.listdir(s_dir) if f.endswith('.dat')]:
                                df = PostProcess.read_zone_dat(os.path.join(s_dir, dat_file_name))
                                if df is not None and not df.empty:
                                    if zone_id not in data_by_zone: data_by_zone[zone_id] = pd.DataFrame()
                                    prefix = f"Zone{zone_id}_{zone_folder.replace(f'Zone_{zone_id}_', '')}_{os.path.splitext(dat_file_name)[0]}_"
                                    df = df.add_prefix(prefix)
                                    data_by_zone[zone_id] = pd.merge(data_by_zone[zone_id], df, left_index=True, right_index=True, how='outer')

                    for zone_id, zone_df in sorted(data_by_zone.items()):
                        all_data_merged = pd.merge(all_data_merged, zone_df, left_index=True, right_index=True, how='outer')

                    if all_data_merged.empty:
                        print(f'Warning: Nenhum dado processado para {sim_dir_name}.')
                        continue

                    all_data_merged.sort_index(inplace=True)
                    all_data_merged.fillna(method='ffill', inplace=True)
                    all_data_merged.fillna(method='bfill', inplace=True)
                    all_data_merged.reset_index(inplace=True)

                    sim_id = int(sim_dir_name.split('_')[1])
                    sim_group_name = f"sim_{str(sim_id).zfill(ndigits)}"
                    sim_group = h5_file.create_group(sim_group_name)
                    sim_group.create_dataset('data', data=all_data_merged.to_records(index=False))
                    print(f'Success: {sim_dir_name} salvo em {sim_group_name}.')

            print(f'\nArquivo HDF5 unificado "{h5_filename}" criado com sucesso.')


if __name__ == '__main__':
    # --- Exemplo de uso simples ---
    # Defina o caminho para a pasta que você quer processar.
    # Pode ser o diretório que contém as pastas 'case_*' (ex: 'P57_results_monitor')
    # ou o caminho direto para uma pasta de caso (ex: 'P57_results_monitor\case_1')
    
    path_para_processar = r'D:\DATA\P66\P66_Sstabdiagram2_scrape'

    if os.path.exists(path_para_processar):
        PostProcess.process_case_to_h5_unified(path_para_processar)
    else:
        print(f"Caminho não encontrado: {path_para_processar}")