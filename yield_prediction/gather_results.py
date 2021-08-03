# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:30:57 2020

@author: alexe
"""
import pandas as pd
from collections import defaultdict
import numpy as np
import os
from itertools import groupby

def group_list(test_list): 
    # Sort list.
    test_list.sort() 
    
    # Group similar substrings.
    grouped_dict = {j: list(i) for j, i in groupby(test_list, 
                      lambda a: a.split('_')[0])}
    
    return grouped_dict

def get_results(descriptor_names, test_types, test_names, sheet_name='scores',
                index_col=0, dir_results='.'):
    results = defaultdict()
    for test_type in test_types:
        results[test_type] = defaultdict()
        
        if test_names[test_type] is not None:
            for test_name in test_names[test_type]:
                results[test_type][test_name] = []
                
                for descriptor_name in descriptor_names:
                    
                    for test_set in os.listdir('{}/results/{}/{}/{}'.format(
                        dir_results, descriptor_name, test_type, test_name
                        )):
                        if os.path.isdir('{}/results/{}/{}/{}/{}'.format(
                            dir_results, descriptor_name, test_type, test_name,
                            test_set
                            )):
                        
                            info = defaultdict()
                            if '/' in descriptor_name:
                                info.update({'Descriptor': descriptor_name.split('/')[1]})
                            else:
                                info.update({'Descriptor': descriptor_name})
                            test = ''.join(
                                [j for j, i in groupby([test_set], lambda a: a.split('_')[0])]
                                )
                            info.update({
                                'Test': test,
                                'Test Set': test_set,
                                })
                            df = pd.read_excel(
                                '{}/results/{}/{}/{}/{}/results.xlsx'.format(
                                    dir_results, descriptor_name, 
                                    test_type, test_name, test_set), 
                                sheet_name=sheet_name, 
                                index_col=index_col
                                )
                            if not df.index.name:
                                df.index.name = sheet_name
                            names = list(info.keys())
                            df = pd.concat(
                                [df], 
                                keys=[tuple(info.values())], 
                                names=names
                                )
                            results[test_type][test_name].append(df)
                            
                            if 'yield_exp' in df.columns:
                                df.drop(columns='yield_exp', inplace=True)
                            
                            df.columns = df.columns.str.lstrip('yield_pred')
                            
                            if descriptor_name == 'graph_descriptors':
                                df.rename(
                                    columns={'SVR - Precomputed Kernel': 'SVR - WL Kernel'}, 
                                    inplace=True
                                    )
                            elif 'SVR - Precomputed Kernel' in df.columns:
                                df.rename(
                                    columns={'SVR - Precomputed Kernel': 'SVR - Tanimoto Kernel'}, 
                                    inplace=True
                                    )
                                
                        elif test_set=='results.xlsx':
                            
                            info = defaultdict()
                            if '/' in descriptor_name:
                                info.update({'Descriptor': descriptor_name.split('/')[1]})
                            else:
                                info.update({'Descriptor': descriptor_name})
                            df = pd.read_excel(
                                '{}/results/{}/{}/{}/results.xlsx'.format(
                                    dir_results, descriptor_name, 
                                    test_type, test_name), 
                                sheet_name=sheet_name, 
                                # index_col=index_col
                                )
                            df.replace(np.nan, 'none', inplace=True)
                            df = df.set_index(['additive', 'aryl_halide', 'base', 'ligand'])
                            if not df.index.name:
                                df.index.name = sheet_name
                            names = list(info.keys())
                            df = pd.concat(
                                [df], 
                                keys=[tuple(info.values())], 
                                names=names
                                )
                            results[test_type][test_name].append(df)
                            
                            if 'yield_exp' in df.columns:
                                df.drop(columns='yield_exp', inplace=True)
                            
                            df.columns = df.columns.str.lstrip('yield_pred')
                            
                            if descriptor_name == 'graph_descriptors':
                                df.rename(
                                    columns={'SVR - Precomputed Kernel': 'SVR - WL Kernel'}, 
                                    inplace=True
                                    )
                            elif 'SVR - Precomputed Kernel' in df.columns:
                                df.rename(
                                    columns={'SVR - Precomputed Kernel': 'SVR - Tanimoto Kernel'}, 
                                    inplace=True
                                    )
                            
                if results[test_type][test_name]:
                    results[test_type][test_name] = pd.concat(
                        results[test_type][test_name]
                        )
                    
                    if any(results[test_type][test_name].index.duplicated()):
                        results[test_type][test_name] = \
                            results[test_type][test_name].sum(
                                axis='index',
                                level=results[test_type][test_name].index.names,
                                min_count=1
                                )
            # else:
            #     info = {
            #         'Descriptor': descriptor_name,
            #         'Test Type': test_type,
            #         }
            #     df = pd.read_excel(
            #         '{}/results/{}/{}/results.xlsx'.format(
            #             dir_results, descriptor_name, test_type), 
            #         sheet_name=sheet_name, 
            #         index_col=index_col
            #         ).T
            #     names = list(info.keys())
            #     names.append('Model')
            #     df = pd.concat(
            #         [df], 
            #         keys=[tuple(info.values())], 
            #         names=names
            #         )
            #     results.append(df)
                
    return results

def get_fp_bit_length_results(scores_mean, metric):
    fps_bit_length = scores_mean.loc[
        (scores_mean.index.isin(['ranking'], level='Test')) 
        & 
        (scores_mean.index.get_level_values(
            'Descriptor').str.contains('Morgan|RDK'))
        ][metric]['Mean']

    fps_bit_length.index = fps_bit_length.index.droplevel('Test')
    # fps_bit_length.columns = fps_bit_length.columns.droplevel(0)
    fps_bit_length = pd.DataFrame(fps_bit_length)
    
    fps_bit_length['Descriptor'] = fps_bit_length.index.get_level_values(
        'Descriptor').str.split('_').str[0].values
    fps_bit_length['Bit Length'] = pd.to_numeric(
        fps_bit_length.index.get_level_values(
            'Descriptor').str.split('_').str[1].values
        )
        
    fps_bit_length.index = fps_bit_length.index.droplevel('Descriptor')
    fps_bit_length = fps_bit_length.reset_index().set_index(
        ['Descriptor', 'Bit Length']
        )
    
    fps_bit_length = fps_bit_length.pivot(columns='Model')
    
    fps_bit_length.columns = fps_bit_length.columns.droplevel(0)
    
    fps_bit_length['Mean'] = fps_bit_length.mean(axis='columns')
    
    return fps_bit_length

'''                    
fps =  [
        'Morgan1_32', 'Morgan1_64', 'Morgan1_128', 'Morgan1_256', 
            'Morgan1_512', 'Morgan1_1024', 'Morgan1_2048',
        'Morgan2_32', 'Morgan2_64', 'Morgan2_128', 'Morgan2_256', 
            'Morgan2_512', 'Morgan2_1024', 'Morgan2_2048',
        'Morgan3_32', 'Morgan3_64', 'Morgan3_128', 'Morgan3_256', 
            'Morgan3_512', 'Morgan3_1024', 'Morgan3_2048',
        'FMorgan1_32', 'FMorgan1_64', 'FMorgan1_128', 'FMorgan1_256',  
            'FMorgan1_512', 'FMorgan1_1024', 'FMorgan1_2048',
        'FMorgan2_32', 'FMorgan2_64', 'FMorgan2_128', 'FMorgan2_256', 
            'FMorgan2_512', 'FMorgan2_1024', 'FMorgan2_2048',
        'FMorgan3_32', 'FMorgan3_64', 'FMorgan3_128', 'FMorgan3_256', 
            'FMorgan3_512', 'FMorgan3_1024', 'FMorgan3_2048',
        'RDK_32', 'RDK_64', 'RDK_128', 'RDK_256', 
            'RDK_512', 'RDK_1024', 'RDK_2048',
        'MACCS', 
        ]
'''  

graphs_folders = [
    'WLlinear_2', 'WLlinear_3', #'WLlinear_4', 'WLlinear_5', 'WLlinear_6', 'WLlinear_7', 'WLlinear_8', 'WLlinear_9', 'WLlinear_10', 
    'WLpolynomial_2', 'WLpolynomial_3', #'WLpolynomial_4', 'WLpolynomial_5', 'WLpolynomial_6', 'WLpolynomial_7', 'WLpolynomial_8', 'WLpolynomial_9', 'WLpolynomial_10',
    'WLsigmoidlogistic_2', 'WLsigmoidlogistic_3', #'WLsigmoidlogistic_4', 'WLsigmoidlogistic_5',
    'WLsigmoidhyperbolictangent_2', 'WLsigmoidhyperbolictangent_3', #'WLsigmoidhyperbolictangent_4', 'WLsigmoidhyperbolictangent_5', 'WLsigmoidhyperbolictangent_6', 'WLsigmoidhyperbolictangent_7', 'WLsigmoidhyperbolictangent_8', 'WLsigmoidhyperbolictangent_9', 'WLsigmoidhyperbolictangent_10',
    'WLsigmoidarctangent_2', 'WLsigmoidarctangent_3', #'WLsigmoidarctangent_4', 'WLsigmoidarctangent_5',
    'WLgaussian_2', 'WLgaussian_3',
    'WLexponential_2', 'WLexponential_3',
    'WLrbf_2', 'WLrbf_3', #'WLrbf_4', 'WLrbf_5', 'WLrbf_6', 'WLrbf_7', 'WLrbf_8', 'WLrbf_9', 'WLrbf_10'
    'WLlaplacian_2', 'WLlaplacian_3',
    'WLmultiquadratic_2', 'WLmultiquadratic_3',
    'WLinversemultiquadratic_2', 'WLinversemultiquadratic_3',
    'WLpower_2', 'WLpower_3',
    'WLlog_2', 'WLlog_3',
    'WLcauchy_2', 'WLcauchy_3'
    ]

dirs = defaultdict()
#dirs['quantum'] = 'quantum_descriptors'
#dirs['quantum_noI'] = 'quantum_descriptors_noI'
#dirs['one-hot'] = 'one_hot_encodings'
for graph in graphs_folders:
    dirs['{}'.format(graph)] = 'graph_descriptors/{}'.format(graph)
#for fp in fps:
#    dirs['{}_raw'.format(fp)] = 'fp_descriptors/{}/raw'.format(fp)
#    dirs['{}_concat'.format(fp)] = 'fp_descriptors/{}/concat'.format(fp)

descriptor_names=[dirs[k] for k in dirs.keys()]
test_types=['out_of_sample']
test_names={
    'out_of_sample': ['additive', 'aryl_halide', 'base', 'ligand'],
    }

scores = get_results(
    descriptor_names=[dirs[k] for k in dirs.keys()],
    test_types=['out_of_sample'],
    test_names={
        'out_of_sample': ['additive', 'aryl_halide', 'base', 'ligand'],
        }
    )

for test_type in scores.keys():
    if test_type == 'out_of_sample':
        scores_mean = defaultdict()
        for test_name in scores[test_type].keys():
            if isinstance(scores[test_type][test_name], pd.DataFrame):
                scores_mean[test_name] = pd.DataFrame()
                
                scores_test_set = scores[test_type][test_name].unstack('Test Set').stack(0)
                
                scores_mean[test_name]['Mean'] = scores_test_set.mean(1)
                scores_mean[test_name]['Std'] = scores_test_set.std(1)
                
                scores_mean[test_name] = scores_mean[test_name].unstack(2)
                scores_mean[test_name] = scores_mean[test_name].reorder_levels(
                    [1,0], axis='columns'
                    ).sort_index(
                        axis='columns', level=[0,1]
                        )
                scores_mean[test_name].index = scores_mean[test_name].index.rename('Model', -1)
                scores_mean[test_name] = scores_mean[test_name].reset_index(
                    ).set_index(['Test', 'Descriptor', 'Model']
                                ).sort_values(['Test', 'Descriptor', 'Model'])
                
                scores[test_type][test_name] = scores[test_type][test_name].unstack().stack(0)
                scores[test_type][test_name].index = scores[test_type][test_name].index.rename('Model', -1)
                scores[test_type][test_name] = scores[test_type][test_name].reset_index(
                    ).set_index(['Test', 'Descriptor', 'Model', 'Test Set']
                                ).sort_values(['Test', 'Descriptor', 'Model', 'Test Set'])

        
        writer = pd.ExcelWriter('results/out_of_sample_results.xlsx')
        for test_name, results in scores[test_type].items():
            results.to_excel(writer, sheet_name=test_name)
        for test_name, results in scores_mean.items():
            results.to_excel(writer, sheet_name='{}_mean'.format(test_name))
        writer.save()

'''
fps_bit_length_results = defaultdict()
fps_bit_length_results['Additive Mean R2'] = get_fp_bit_length_results(
    scores_mean['additive'], 
    'R-squared'
    )
fps_bit_length_results['Additive Mean RMSE'] = get_fp_bit_length_results(
    scores_mean['additive'], 
    'RMSE'
    )
fps_bit_length_results['Aryl Halide Mean R2'] = get_fp_bit_length_results(
    scores_mean['aryl_halide'], 
    'R-squared'
    )
fps_bit_length_results['Aryl Halide Mean RMSE'] = get_fp_bit_length_results(
    scores_mean['aryl_halide'], 
    'RMSE'
    )

writer = pd.ExcelWriter('results/fp_bit_length_results.xlsx', engine='xlsxwriter')
for name, results in fps_bit_length_results.items():
    results.to_excel(writer, sheet_name=name)
writer.save()
'''
 

'''
descriptor_names=[
    'quantum_descriptors', 
    'fp_descriptors/Morgan1_1024/raw', 'fp_descriptors/Morgan1_1024/concat',
    'fp_descriptors/Morgan2_1024/raw', 'fp_descriptors/Morgan2_1024/concat',
    'fp_descriptors/Morgan3_1024/raw', 'fp_descriptors/Morgan3_1024/concat',
    'fp_descriptors/MACCS/raw', 'fp_descriptors/MACCS/concat',
    'fp_descriptors/RDK_1024/raw', 'fp_descriptors/RDK_1024/concat',
    'graph_descriptors', 'one_hot_encodings'
    ]
'''

descriptor_names=[
    'graph_descriptors/WLlinear_2', 'graph_descriptors/WLlinear_3', #'graph_descriptors/WLlinear_4', 'graph_descriptors/WLlinear_5', 'graph_descriptors/WLlinear_6', 'graph_descriptors/WLlinear_7', 'graph_descriptors/WLlinear_8', 'graph_descriptors/WLlinear_9', 'graph_descriptors/WLlinear_10',
    'graph_descriptors/WLpolynomial_2', 'graph_descriptors/WLpolynomial_3', #'graph_descriptors/WLpolynomial_4', 'graph_descriptors/WLpolynomial_5', 'graph_descriptors/WLpolynomial_6', 'graph_descriptors/WLpolynomial_7', 'graph_descriptors/WLpolynomial_8', 'graph_descriptors/WLpolynomial_9', 'graph_descriptors/WLpolynomial_10',
    'graph_descriptors/WLsigmoidlogistic_2', 'graph_descriptors/WLsigmoidlogistic_3', #'graph_descriptors/WLsigmoidlogistic_4', 'graph_descriptors/WLsigmoidlogistic_5',
    'graph_descriptors/WLsigmoidhyperbolictangent_2', 'graph_descriptors/WLsigmoidhyperbolictangent_3', #'graph_descriptors/WLsigmoidhyperbolictangent_4', 'graph_descriptors/WLsigmoidhyperbolictangent_5', 'graph_descriptors/WLsigmoidhyperbolictangent_6', 'graph_descriptors/WLsigmoidhyperbolictangent_7', 'graph_descriptors/WLsigmoidhyperbolictangent_8', 'graph_descriptors/WLsigmoidhyperbolictangent_9', 'graph_descriptors/WLsigmoidhyperbolictangent_10',
    'graph_descriptors/WLsigmoidarctangent_2', 'graph_descriptors/WLsigmoidarctangent_3', #'graph_descriptors/WLsigmoidarctangent_4', 'graph_descriptors/WLsigmoidarctangent_5',
    'graph_descriptors/WLgaussian_2', 'graph_descriptors/WLgaussian_3',
    'graph_descriptors/WLexponential_2', 'graph_descriptors/WLexponential_3',
    'graph_descriptors/WLrbf_2', 'graph_descriptors/WLrbf_3', #'graph_descriptors/WLrbf_4', 'graph_descriptors/WLrbf_5', 'graph_descriptors/WLrbf_6', 'graph_descriptors/WLrbf_7', 'graph_descriptors/WLrbf_8', 'graph_descriptors/WLrbf_9', 'graph_descriptors/WLrbf_10'
    'graph_descriptors/WLlaplacian_2', 'graph_descriptors/WLlaplacian_3',
    'graph_descriptors/WLmultiquadratic_2', 'graph_descriptors/WLmultiquadratic_3',
    'graph_descriptors/WLinversemultiquadratic_2', 'graph_descriptors/WLinversemultiquadratic_3',
    'graph_descriptors/WLpower_2', 'graph_descriptors/WLpower_3',
    'graph_descriptors/WLlog_2', 'graph_descriptors/WLlog_3',
    'graph_descriptors/WLcauchy_2', 'graph_descriptors/WLcauchy_3',
    ]

test_types=['out_of_sample']
test_names={
    'out_of_sample': ['additive', 'aryl_halide', 'base', 'ligand'],
    }
test_names={
        'out_of_sample': ['additive', 'aryl_halide'],
        }

y_pred = get_results(
        descriptor_names=descriptor_names,
        test_types=['out_of_sample'],
        test_names={
            'out_of_sample': ['additive', 'aryl_halide', 'base', 'ligand'],
            },
        sheet_name='y_pred',
        index_col=[0,1,2,3]
       )

for test_type in y_pred.keys():
    if test_type == 'out_of_sample':
        y_pred_to_save = defaultdict(dict)

        for test_name in y_pred[test_type].keys():
            descriptors = y_pred[test_type][test_name].index.get_level_values(
                'Descriptor').drop_duplicates()
            
            for descriptor in descriptors:
                if 'ranking' in y_pred[test_type][test_name].index.get_level_values(
                        'Test'):
                    y_pred_to_save[test_name][descriptor] = y_pred[test_type][test_name][
                        (y_pred[test_type][test_name].index.get_level_values(
                            'Descriptor') == descriptor)
                        &
                        (y_pred[test_type][test_name].index.get_level_values(
                            'Test') == 'ranking')
                        ].dropna(axis='columns', how='all')
                elif 'LOO' in y_pred[test_type][test_name].index.get_level_values(
                        'Test'):
                    y_pred_to_save[test_name][descriptor] = y_pred[test_type][test_name][
                        (y_pred[test_type][test_name].index.get_level_values(
                            'Descriptor') == descriptor)
                        &
                        (y_pred[test_type][test_name].index.get_level_values(
                            'Test') == 'LOO')
                        ].dropna(axis='columns', how='all')
                
                y_pred_to_save[test_name][descriptor].index = \
                    y_pred_to_save[test_name][descriptor].index.droplevel(
                        ['Descriptor', 'Test'])                

for name, fps in y_pred_to_save.items():
    writer = pd.ExcelWriter('results/main_text_{}_y_pred.xlsx'.format(name),)# engine='xlsxwriter')
    for k, df in fps.items():
        df.to_excel(writer, sheet_name='{}'.format(k))
    writer.save()
           
     

for test_type in scores.keys():
    if test_type == 'out_of_sample':
        descriptors = []
        for descriptor in descriptor_names:
            if '/' in descriptor:
                descriptors.append(descriptor.split('/')[1])
            else:
                descriptors.append(descriptor)
        
        scores_subset = defaultdict()
        for test_name in test_names[test_type]:
            
            scores_subset[test_name] = pd.concat([
                 scores[test_type][test_name][
                    (scores[test_type][test_name].index.isin(
                        descriptors, level='Descriptor'))
                    &
                    (scores[test_type][test_name].index.get_level_values(
                        'Test') == 'ranking')
                    ].unstack('Test Set'),
                scores_mean[test_name][
                    (scores_mean[test_name].index.isin(
                        descriptors, level='Descriptor'))
                    &
                    (scores_mean[test_name].index.get_level_values(
                        'Test') == 'ranking')
                    ]
                ], axis=1)
            scores_subset[test_name] = scores_subset[test_name].sort_values(
                ['scores'], 
                axis='columns'
                )            
    
        writer = pd.ExcelWriter('results/main_text_scores.xlsx')
        for test_name, results in scores_subset.items():
            results.to_excel(writer, sheet_name=test_name)
        writer.save()

'''       
descriptor_names=[
    'quantum_descriptors_missing_additive', 
    'fp_descriptors/Morgan1_1024/raw', 'fp_descriptors/Morgan1_1024/concat',
    'graph_descriptors', 'one_hot_encodings'
    ]

test_types=['validation']
test_names={
    'validation': ['subset_mols'],
    }

y_pred_prospective = get_results(
        descriptor_names=descriptor_names,
        test_types=test_types,
        test_names=test_names,
        sheet_name='y_pred',
        index_col=[0,1,2,3]
       )

y_pred_to_save = defaultdict(dict)

for test_type in test_types:
    for test_name in y_pred_prospective[test_type].keys():
        descriptors = y_pred_prospective[test_type][test_name].index.get_level_values(
            'Descriptor').drop_duplicates()
        
        for descriptor in descriptors:
            descriptor_name = '_'.join(descriptor.split('_')[0:2])
            y_pred_to_save[test_name][descriptor_name] = y_pred_prospective[test_type][test_name][
                (y_pred_prospective[test_type][test_name].index.get_level_values(
                    'Descriptor') == descriptor)
                ].dropna(axis='columns', how='all')
            
            y_pred_to_save[test_name][descriptor_name].index = \
                y_pred_to_save[test_name][descriptor_name].index.droplevel(
                    ['Descriptor'])

        writer = pd.ExcelWriter('results/prospective_ypred_subsetmols.xlsx')
        for descriptor, results in y_pred_to_save[test_name].items():
            results.to_excel(writer, sheet_name=descriptor, merge_cells=False)
        writer.save()

descriptor_names=[
    'fp_descriptors/Morgan1_1024/raw', 'fp_descriptors/Morgan1_1024/concat',
    'graph_descriptors', 'one_hot_encodings'
    ]
test_types=['validation']
test_names={
    'validation': ['all_mols'],
    }

y_pred_prospective = get_results(
        descriptor_names=descriptor_names,
        test_types=test_types,
        test_names=test_names,
        sheet_name='y_pred',
        index_col=[0,1,2,3]
       )

y_pred_to_save = defaultdict(dict)

for test_type in test_types:
    for test_name in y_pred_prospective[test_type].keys():
        descriptors = y_pred_prospective[test_type][test_name].index.get_level_values(
            'Descriptor').drop_duplicates()
        
        for descriptor in descriptors:
            descriptor_name = '_'.join(descriptor.split('_')[0:2])
            y_pred_to_save[test_name][descriptor_name] = y_pred_prospective[test_type][test_name][
                (y_pred_prospective[test_type][test_name].index.get_level_values(
                    'Descriptor') == descriptor)
                ].dropna(axis='columns', how='all')
            
            y_pred_to_save[test_name][descriptor_name].index = \
                y_pred_to_save[test_name][descriptor_name].index.droplevel(
                    ['Descriptor'])
        
        writer = pd.ExcelWriter('results/prospective_ypred_allmols.xlsx')
        for descriptor, results in y_pred_to_save[test_name].items():
            results.to_excel(writer, sheet_name=descriptor, merge_cells=False)
        writer.save()
'''         
