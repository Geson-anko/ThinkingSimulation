from attrdict import AttrDict
import json
def load_setting(setting_file) -> AttrDict:
    with open(setting_file,"r",encoding="utf-8") as f:
        d = json.load(f)
    setting = AttrDict(d)
    return setting