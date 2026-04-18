import re
import time

class Prefix(object):
    pattern_en_value_type = ["http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2001/XMLSchema#",
                             "http://dbpedia.org/datatype/"]

    @classmethod
    def set_language(cls, language):
        if language == 'en':
            cls.regex_ent = re.compile(r'http:\/\/dbpedia\.org\/resource\/(.*)')
            cls.pattern_prop = 'http://dbpedia.org/property/'
        elif language in {'zh', 'fr', 'ja'}:
            cls.regex_ent = re.compile(r'http:\/\/%s\.dbpedia\.org\/resource\/(.*)' % language)
            cls.pattern_prop = 'http://%s.dbpedia.org/property/' % language
        else:
            raise Exception()

    @classmethod
    def remove_prefix(cls, input):
        if isinstance(input, str):
            input = cls.regex_ent.match(input).group(1)
            return input.replace('_', ' ')
        return [cls.remove_prefix(item) for item in input]

    @classmethod
    def remove_prop_prefix(cls, input):
        if isinstance(input, str):
            if input.find(cls.pattern_prop) >= 0:
                return input.split(cls.pattern_prop)[1]
            raise Exception()
        return [cls.remove_prop_prefix(item) for item in input]

    @classmethod
    def remove_value_type(cls, input):
        if isinstance(input, str):
            for pattern in cls.pattern_en_value_type:
                if input.find(pattern) >= 0:
                    return input.split(pattern)[1]
            raise Exception()
        return [cls.remove_value_type(item) for item in input]


def read_file(path, parse_func):
    num = -1
    with open(path, 'r', encoding='utf8') as f:
        line = f.readline().strip()
        if line.isdigit():
            num = int(line)
        else:
            f.seek(0)
        lines = f.readlines()

    lines = parse_func(lines)

    if len(lines) != num and num >= 0:
        raise ValueError()
    return lines


def read_triples(path):
    return read_file(path, lambda lines: [tuple([int(item) for item in line.strip().split('\t')]) for line in lines])


def read_mapping(path):
    def _parser(lines):
        for idx, line in enumerate(lines):
            i, name = line.strip().split('\t')
            lines[idx] = (int(i), name)
        return dict(lines)
    return read_file(path, _parser)


def load_language(directory,flag):
    id2entity = read_mapping(directory + '/ent_ids_' + flag )
    return id2entity



def load_dbpedia_properties(data_path, entity2id, filter_alias=False):
    potential_alias_pattern = ['name', 'alias', '名字', '别名']
    with open(data_path, 'r', encoding='utf8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    att_triples = []
    for line in lines:
        try:
            subject, property, value, _ = line
        except ValueError:
            subject, property, value = line

        # filter the alias
        if filter_alias:
            for alias in potential_alias_pattern:
                if property.lower().find(alias) >= 0:
                    value = ''
        try:
            value = value.encode('utf8').decode('unicode_escape')
        except UnicodeDecodeError:
            pass
        
        ent_id = entity2id[subject]
        att = property
        att_triples.append((ent_id, att, value))
   
    return att_triples


def gat_att_id(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip().split("\n")
        data = [i.split("\t") for i in data]  # 根据换行符进行拆分，生成一个由多个子字符串组成的列表"data"，每个子字符串都表示一个实体和其对应的ID值。
        ids.append(set([int(i[0]) for i in data]))  # 提取出实体的id值，放入列表
        return ids[0]

def get_quanzhong(index_a,index_b,all):
    index = index_a + index_b
    total_count = len(index)
    # 生成新的列表
    result = [index.count(a) / total_count for a in all]
    return result

def get_all_atts(args):
    # 加载源语言和目标语言的实体映射
    id2entity_sr = load_language(f'{args.dataset_folder}/{args.dataset}/{args.language}', '1')
    id2entity_tg = load_language(f'{args.dataset_folder}/{args.dataset}/{args.language}', '2')
    # 构建实体到ID的映射
    entity2id_sr = {ent: idx for idx, ent in id2entity_sr.items()}
    entity2id_tg = {ent: idx for idx, ent in id2entity_tg.items()}

    # 根据数据集类型确定属性文件名格式
    if args.dataset == "DWY100K":
        # DWY100K数据集使用不同的属性文件命名方式
        if args.language == "wd_dbp":
            att_file_1 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_wd.txt'
            att_file_2 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_dbp.txt'
        elif args.language == "yg_dbp":
            # yg_dbp使用atts_properties_1.txt和atts_properties_2.txt
            att_file_1 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_yg.txt'
            att_file_2 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_dbp.txt'
        else:
            # 默认处理方式，适用于其他DWY100K变体
            parts = args.language.split('_')
            if len(parts) >= 2:
                lang1, lang2 = parts[0], parts[1]
                att_file_1 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_{lang1}.txt'
                att_file_2 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_{lang2}.txt'
            else:
                # 如果格式不符合预期，使用默认方式
                att_file_1 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_{args.language[0:2]}.txt'
                att_file_2 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_{args.language[3:5]}.txt'
    else:
        # 对于DBP15K和SRPRS数据集，使用原有逻辑
        att_file_1 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_{args.language[0:2]}.txt'
        att_file_2 = f'{args.dataset_folder}/{args.dataset}/{args.language}/atts_properties_{args.language[3:5]}.txt'

    # 加载属性三元组并转换为ID形式
    att_triples_zh = load_dbpedia_properties(att_file_1, entity2id_sr)
    att_triples_en = load_dbpedia_properties(att_file_2, entity2id_tg)

    # 加载属性到ID的映射
    id2atts = read_mapping(f'{args.dataset_folder}/{args.dataset}/{args.language}/id2atts.txt')
    att2id = {att: idx for idx, att in id2atts.items()}

    # 获取属性ID集合
    att_id = gat_att_id(f'{args.dataset_folder}/{args.dataset}/{args.language}/id2atts.txt')

    # 将属性三元组中的属性名称转换为属性ID
    att_triples_zh = [(ent_id, att2id[att_key], att_value) for ent_id, att_key, att_value in att_triples_zh if att_key in att2id]
    att_triples_en = [(ent_id, att2id[att_key], att_value) for ent_id, att_key, att_value in att_triples_en if att_key in att2id]

    # 提取属性键和属性值
    att_value_zh = [att_value for _, _, att_value in att_triples_zh]
    att_value_en = [att_value for _, _, att_value in att_triples_en]


    # 返回处理后的数据
    return att_triples_zh, att_triples_en, att_value_zh, att_value_en, att_id # att_weight


