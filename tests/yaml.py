from ruamel.yaml.constructor import DuplicateKeyError

from prepare_quran_dataset.construct.utils import load_yaml, dump_yaml

if __name__ == '__main__':

    yaml_str = """
    001: http://example.com
    002: http://example.com
    03: http://example.com
    """

    # yaml_str = """
    # - http:///
    # - http:///hajmo
    # """

    out_dict = load_yaml(yaml_str)
    print(out_dict)

    d = {1: 'hamo', 2: 'hamo'}
    print(dump_yaml(out_dict))
