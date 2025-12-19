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
        Algoritmo pode ser utilizado para a leitura dos seguintes arquivos de
        pos-processamento gerados pelo openfoam:

        force.dat, moment.dat, residuals.dat, continuityError.dat

        Parameters
        ----------
        file : Caminho para arquivo force/moment.dat ou residuals.dat

        Returns
        -------
        data : dataframe com os dados dentro do arquivo .dat
        """

        with open(file) as filename:
            logFile = filename.readlines()
        if not logFile:
            return
        data = []
        flag_header = False
        for idx, line in enumerate(logFile):
            l = list(filter(None, [re.sub('[()#]', '', group) for group
                                   in line.split()]))

            if not l:
                continue
            if not flag_header:
                if l[0] != 'Time':
                    continue
                flag_header = True
            data.append(l)

        data = pd.DataFrame(data)
        data = data.rename(columns=data.iloc[0]).drop(
            data.index[0]).apply(pd.to_numeric, errors='ignore')
        data = data.set_index('Time')
        return data
    
    @staticmethod
    def convert_dat_h5(fname_force, fname_moment, fname_h5):
        data_force = PostProcess.read_DAT(fname_force)
        data_moment = PostProcess.read_DAT(fname_moment)

        data_force = data_force.rename(columns={
            'total_x':'FX0',
            'total_y':'FY0',
            'total_z':'FZ0'
        })
        data_moment = data_moment.rename(columns={
            'total_x':'MX0',
            'total_y':'MY0',
            'total_z':'MZ0'
        })

        data_force.index = data_force.index.astype(float)
        data_moment.index = data_moment.index.astype(float)

        data_juncao = pd.merge(
            data_force[['FX0','FY0','FZ0']],
            data_moment[['MX0','MY0','MZ0']],
            right_index = True,
            left_index = True
        )

        compound_dtype = np.dtype([
            ('Time', 'f8'),
            ('FX0' , 'f8'),
            ('FY0' , 'f8'),
            ('FZ0' , 'f8'),
            ('MX0' , 'f8'),
            ('MY0' , 'f8'),
            ('MZ0' , 'f8') 
        ])
        compound_dtype = data_juncao.to_records()
        
        with h5py.File(fname_h5, 'w') as h5:
            h5.create_dataset('force_moment_data', data=compound_dtype)


    @staticmethod
    def process_case_to_h5(case_path):
        """
       	Algorítmo pode ser utilizado para percorrer uma pasta case, encontrar todas as subpastas de simulação (sim_1, sim_2, ...),
        e processar automaticamente todos os arquivos .dat encontrados dentro de seus respectivos
        diretórios 'postProcessing'.

	    case_1.h5

        Parameters
        ----------
        case_path: O caminho para a pasta do caso principal que contém as subpastas 'sim_*'.

        Outputs
        ----------
        file: A função cria um arquivo .h5.
        """

        case_name = os.path.basename(os.path.normpath(case_path))
        parent_dir = os.path.dirname(case_path)
        h5_filename = os.path.join(parent_dir, f'{case_name}.h5')

        with h5py.File(h5_filename, 'w') as h5_file:
            sim_dirs_names = []
            sim_num = []

            for d in os.listdir(case_path):
                if d.startswith('sim_') and os.path.isdir(os.path.join(case_path, d)):
                    sim_dirs_names.append(d)

            sim_dirs = sorted(sim_dirs_names, key=lambda x: int(x.split('_')[1]))

            if sim_dirs:
                for d in sim_dirs:
                    separa = d.split('_')
                    num = int(separa[1])
                    sim_num.append(num)
                    max_sim = max(sim_num)
                if max_sim > 0:
                    ndigits = math.floor(math.log10(max_sim))+ 1
                else:
                    ndigits = 1


            for sim_dirs_name in sim_dirs:
                print(f'--- Processing {sim_dirs_name} ---')
                sim_path = os.path.join(case_path, sim_dirs_name)
                post_processing_path = os.path.join(sim_path, 'postProcessing')

                if not os.path.isdir(post_processing_path):
                    continue

                all_data_merged = pd.DataFrame()

                data_folders = [d for d in os.listdir(post_processing_path) if os.path.isdir(os.path.join(post_processing_path, d))]

                for folder in data_folders:
                    data_dir = os.path.join(post_processing_path, folder, '0')
                    if not os.path.isdir(data_dir):
                        continue

                    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
                    for dat_file in dat_files:
                        file_path = os.path.join(data_dir, dat_file)
                        data = PostProcess.read_DAT(file_path)

                        if data is None or data.empty:
                            continue
                        
                        if dat_file == 'solverInfo.dat':
                            cols_to_keep = [c for c in data.columns if c.endswith('_final')]
                            data = data[cols_to_keep]
                        else:
                            data = data.select_dtypes(include=np.number)

                        if data.empty: continue

                        prefix = f"{dat_file.replace('.dat', '')}_"
                        data = data.add_prefix(prefix)

                        if all_data_merged.empty:
                            all_data_merged = data
                        else:
                            if not isinstance(data.index, pd.RangeIndex) or not isinstance(all_data_merged.index, pd.RangeIndex):
                                all_data_merged = pd.merge(all_data_merged, data, left_index=True, right_index=True, how='outer')
                            else: 
                                for col in data.columns:
                                    all_data_merged[col] = data[col].iloc[0]


                if all_data_merged.empty:
                    print(f'Warning: No data could be processed for {sim_dirs_name}.')
                    continue
                
                all_data_merged.fillna(method='ffill', inplace=True)
                all_data_merged.fillna(method='bfill', inplace=True)

                compound_data = all_data_merged.to_records()

                sim_id = int(sim_dirs_name.split('_')[1])
                new_name = f'sim_{str(sim_id).zfill(ndigits)}'

                sim_group = h5_file.create_group(new_name)
                sim_group.create_dataset(new_name, data=compound_data)

                print(f'Success: {sim_dirs_name} data saved.')
        
        print(f'\nArquivo HDF5 "{h5_filename}" criado com sucesso.\n')


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
        
        if header_line_index == -1: return None

        header_fields = lines[header_line_index].strip().replace('#', '').strip().split()
        
        data_rows = []
        for line in lines[header_line_index + 1:]:
            if line.strip() and not line.strip().startswith('#'):
                cleaned_line = re.sub(r'[()]', '', line)
                data_rows.append(cleaned_line.strip().split())
        
        if not data_rows: return None

        num_data_cols = len(data_rows[0])
        
        # Constrói o cabeçalho expandido
        expanded_header = []
        col_cursor = 0
        field_cursor = 0
        while field_cursor < len(header_fields):
            field_name = header_fields[field_cursor]
            
            # O primeiro campo é sempre 'Time'
            if field_name == 'Time':
                expanded_header.append('Time')
                col_cursor += 1
                field_cursor += 1
                continue
            
            is_vector = '(U)' in field_name and (col_cursor + 2 < num_data_cols)

            # Heurística: Se um campo começa com '(' ou é um vetor conhecido
            # e ainda há pelo menos 3 colunas de dados restantes, trate como vetor.
            if is_vector:
                expanded_header.extend([f"{field_name}_x", f"{field_name}_y", f"{field_name}_z"])
                col_cursor += 3
            else: # Trata como escalar
                expanded_header.append(field_name)
                col_cursor += 1
            field_cursor += 1
        
        # Garante que o número de colunas do cabeçalho e dos dados coincida
        if len(expanded_header) != num_data_cols:
            # Se a heurística falhar, volte para um método simples
            # para evitar que o programa quebre.
            return PostProcess.read_DAT(file)

        df = pd.DataFrame(data_rows, columns=expanded_header)
        df = df.apply(pd.to_numeric, errors='coerce').set_index('Time')
        
        return df


    @staticmethod
    def zone_to_h5(case_path):
        """
        Percorre uma pasta de caso, encontra subpastas de simulação (sim_1, sim_2, ...),
        e processa arquivos .dat de diretórios de zona (Zone_x_*_max)
        dentro de 'postProcessing'. Os dados de cada zona principal (Zone_0, Zone_1, etc.)
        são agrupados e salvos em subgrupos correspondentes no arquivo HDF5.

        Parameters
        ----------
        case_path: O caminho para a pasta do caso principal que contém as subpastas 'sim_*'.

        Outputs
        ----------
        file: Cria um arquivo .h5 com o sufixo '_zones'.
        """
        case_name = os.path.basename(os.path.normpath(case_path))
        parent_dir = os.path.dirname(case_path)
        h5_filename = os.path.join(parent_dir, f'{case_name}_zones.h5')

        with h5py.File(h5_filename, 'w') as h5_file:
            sim_dirs_names = [d for d in os.listdir(case_path) if d.startswith('sim_') and os.path.isdir(os.path.join(case_path, d))]
            sim_dirs = sorted(sim_dirs_names, key=lambda x: int(x.split('_')[1]))

            if not sim_dirs:
                print(f"Nenhuma pasta 'sim_*' encontrada em {case_path}")
                return

            sim_numbers = [int(d.split('_')[1]) for d in sim_dirs]
            max_sim_number = max(sim_numbers)
            ndigits = math.floor(math.log10(max_sim_number)) + 1 if max_sim_number > 0 else 1

            for sim_dir_name in sim_dirs:
                print(f'--- Processing {sim_dir_name} ---')
                sim_path = os.path.join(case_path, sim_dir_name)
                post_processing_path = os.path.join(sim_path, 'postProcessing')

                if not os.path.isdir(post_processing_path):
                    continue

                data_by_zone = {}
                zone_pattern = re.compile(r'^Zone_(\d+)_.*_max$')

                all_zone_folders = [d for d in os.listdir(post_processing_path) if zone_pattern.match(d) and os.path.isdir(os.path.join(post_processing_path, d))]

                for zone_folder in sorted(all_zone_folders):
                    match = zone_pattern.match(zone_folder)
                    if not match:
                        continue
                    
                    zone_id = int(match.group(1))

                    if zone_id not in data_by_zone:
                        data_by_zone[zone_id] = pd.DataFrame()

                    data_dir = os.path.join(post_processing_path, zone_folder, '0')
                    if not os.path.isdir(data_dir):
                        continue

                    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
                    for dat_file in dat_files:
                        file_path = os.path.join(data_dir, dat_file)
                        data = PostProcess.read_DAT(file_path)

                        if data is None or data.empty:
                            continue
                        
                        data = data.select_dtypes(include=np.number)
                        if data.empty:
                            continue

                        prefix = f"{zone_folder}_{dat_file.replace('.dat', '')}_"
                        data = data.add_prefix(prefix)

                        if data_by_zone[zone_id].empty:
                            data_by_zone[zone_id] = data
                        else:
                            data_by_zone[zone_id] = pd.merge(data_by_zone[zone_id], data, left_index=True, right_index=True, how='outer')

                if not data_by_zone:
                    print(f'Warning: No zone data could be processed for {sim_dir_name}.')
                    continue
                
                sim_id = int(sim_dir_name.split('_')[1])
                sim_group_name = f"sim_{str(sim_id).zfill(ndigits)}"
                sim_group = h5_file.create_group(sim_group_name)

                for zone_id, zone_data_merged in sorted(data_by_zone.items()):
                    if zone_data_merged.empty:
                        continue

                    zone_data_merged.fillna(method='ffill', inplace=True)
                    zone_data_merged.fillna(method='bfill', inplace=True)

                    df_final = zone_data_merged.reset_index()
                    
                    duplicated = df_final.columns.duplicated()
                    if duplicated.any():
                        df_final = df_final.loc[:, ~duplicated]

                    compound_data = df_final.to_records(index=False)
                    
                    dataset_name = f'zone_{zone_id}'
                    sim_group.create_dataset(dataset_name,data=compound_data)
        

        print(f'\nArquivo HDF5 de zonas "{h5_filename}" criado com sucesso.\n')    


    @staticmethod
    def read_areaAverage(file, fields=''):
        """
        Algoritmo pode ser utilizado para a leitura dos seguintes arquivos de
        pos-processamento gerados pelo openfoam:

        surfaceFieldValue.dat

        Parameters
        ----------
        file : Caminho para arquivo force/moment.dat ou residuals.dat

        fields : string com os campos que foram avaliados pelo OF separados
        por espaco. Exemplo: 'p U'

        Returns
        -------
        data : dataframe com os dados dentro do arquivo .dat
        """
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
    def plot_data(data, x, y, title='', ylabel='', xlabel='', logy=False):
        if x == 'index':
            plot = data.plot(y=y, title=title, logy=logy)
        else:
            plot = data.plot(x=x, y=y, title=title, xlabel=xlabel, logy=logy)
            plot.set_xlabel(xlabel)
        plot.set_ylabel(ylabel)
        return plot


if __name__ == '__main__':
    # Exemplo de como usar a nova função:
    # Substitua 'caminho/para/sua/pasta/case_1' pelo caminho real da sua pasta.
    # Por exemplo: 'd:\\DATA\\P66_Sstabdiagram2_scrape\\case_1'
    
    # path_para_o_caso = 'd:\\DATA\\P66_Sstabdiagram2_scrape\\case_1'
    # if os.path.exists(path_para_o_caso):
    #     PostProcess.process_case_to_h5(path_para_o_caso)
    # else:
    #     print(f"Caminho não encontrado: {path_para_o_caso}")
    pass

# TESTANDO
path_para_caso = r'D:\DATA\P57\P57_results_monitor\case_1'

if os.path.exists(path_para_caso):
    PostProcess.zone_to_h5(path_para_caso)
else:
    print(f"path not found: {path_para_caso}")