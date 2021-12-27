def get_categ_index_mapping(df_train, categ_feature):
    
    base_mapping = dict(zip(df_train[categ_feature], df_train[categ_feature].cat.codes + 1))
    return {**{"N/A": 0}, **base_mapping}

def get_categs_mappings(df_train, categs):
    mappings_list = []
    for i, c in enumerate(categs):
        map_dict = get_categ_index_mapping(df_train, c)
        mappings_list.append({'col': i, 'mapping': map_dict})
    return mappings_list

def get_emb_size(n_cat):
    return min(10, round(1.6 * n_cat**0.56))
