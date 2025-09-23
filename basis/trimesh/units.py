from .constants import log

# 转换为英寸数
_TO_INCHES = {'microinches': 1.0 / 1000.0,
              'mils': 1.0 / 1000.0,
              'inches': 1.00,
              'feet': 12.0,
              'yards': 36.0,
              'miles': 63360,
              'angstroms': 1.0 / 2.54e8,
              'nanometers': 1.0 / 2.54e7,
              'microns': 1.0 / 2.54e4,
              'millimeters': 1.0 / 2.54e1,
              'centimeters': 1.0 / 2.54e0,
              'meters': 1.0 / 2.54e-2,
              'kilometers': 1.0 / 2.54e-5,
              'decimeters': 1.0 / 2.54e-1,
              'decameters': 1.0 / 2.54e-3,
              'hectometers': 1.0 / 2.54e-4,
              'gigameters': 1.0 / 2.54e-11,
              'AU': 5889679948818.897,
              'light years': 3.72461748e17,
              'parsecs': 1.21483369e18}

# 如果一个单位被其他符号所知道,请将它们包含在此处
_synonyms = {'millimeters': ['mm'],
             'inches': ['in'],
             'meters': ['m']}

for key, new_keys in _synonyms.items():
    _value = _TO_INCHES[key]
    for new_key in new_keys:
        _TO_INCHES[new_key] = _value


def unit_conversion(current, desired):
    '''
    计算从一个单位系统到另一个单位系统的转换

    :param current: str,当前单位系统的名称(例如 'millimeters')
    :param desired: str,目标单位系统的名称(例如 'inches')
    :return: float,转换因子,用于将值转换为目标单位
    '''
    conversion = _TO_INCHES[current] / _TO_INCHES[desired]
    return conversion


def validate(units):
    '''
    检查字符串是否表示有效单位的名称

    :param units: str,表示单位名称的字符串
    :return: bool,指示单位字符串是否为有效单位
    '''
    valid = str(units) in _TO_INCHES
    return valid


def unit_guess(scale):
    '''
    根据比例对图纸或模型的单位进行猜测

    :param scale: float,表示图纸或模型的比例
    :return: str,猜测的单位名称(例如 'millimeters' 或 'inches')
    '''
    if scale > 100.0:
        return 'millimeters'
    else:
        return 'inches'


def _set_units(obj, desired, guess):
    '''
    给定一个具有单位和顶点属性的对象,转换其单位

    :param obj: objects,具有单位和顶点属性的对象(例如 Path 或 Trimesh)
    :param desired: str,目标单位名称(例如 'inches')
    :param guess: bool,是否允许猜测文档的单位如果未指定
    :return: None,函数直接修改对象的单位和顶点属性
    '''
    desired = str(desired)
    if not validate(desired):
        raise ValueError(desired + ' 不是一个有效单位!')

    if obj.units is None:
        if guess:
            obj.units = unit_guess(obj.scale)
            log.warn('没有指定单位,应该是当前单位 %s', obj.units)
        else:
            raise ValueError('没有指定单位,不允许猜测!')

    log.info('转换单位 from %s to %s', obj.units, desired)
    conversion = unit_conversion(obj.units, desired)
    obj.vertices *= conversion
    obj.units = desired
