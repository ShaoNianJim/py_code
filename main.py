# import package
import numpy as np
from pandas import DataFrame
import pandas as pd
import re
from dateutil import relativedelta
import datetime as dt


# 1.1
def df_groupby(df, groupkey, col, func, res_col_name, asint=False, dup=False):
    """
    :param df: 一个df   征对 1+ 用户
    :param groupkey: df中聚合分类的变量名
    :param col:  df中待聚合的变量名，字符串或者列表
    :param func: 聚合方式，支持sum /max /min /avg /count/ distinct_count
    :param res_col_name:  聚合结果列名，字符串或者列表
    :param asint: if asint=True ,聚合结果转为int ;default asint=False;
    :param dup: if dup=True ,变量取值去重 ;default dup=False;
    :return:df_res  df
    """
    # dropna  all row
    df = df.dropna(axis=0, how='all')
    #  reformat type
    try:
        if func != 'count' and func != 'distinct_count':
            df[col] = df[col].astype('float32')
    except ValueError:
        print('the col could not convert string to float!')
    # duplicate the col
    if dup:
        df = df.drop_duplicates(df.columns)
    # compatible str
    if type(col) != list:
        col = [col]
    if type(res_col_name) != list:
        res_col_name = [res_col_name]
    if type(func) != list:
        func = [func]
    # agg index
    df_res = DataFrame(df[groupkey].unique(), columns=[groupkey])
    for i in func:
        if i == 'sum':
            df_res_ago = DataFrame(df.groupby(groupkey)[col].sum())
        elif i == 'max':
            df_res_ago = DataFrame(df.groupby(groupkey)[col].max())
        elif i == 'min':
            df_res_ago = DataFrame(df.groupby(groupkey)[col].min())
        elif i == 'avg':
            df_res_ago = DataFrame(df.groupby(groupkey)[col].mean())
        elif i == 'std':
            df_res_ago = DataFrame(df.groupby(groupkey)[col].std())
        elif i == 'count':
            df_res_ago = DataFrame(df.groupby(groupkey)[col].count())
        elif i == 'distinct_count':
            df_res_ago = DataFrame(df.groupby(groupkey)[col].nunique())
        else:
            print('input func error!')
        df_res_ago = df_res_ago.reset_index()
        df_res = pd.merge(df_res, df_res_ago, how='left', on=groupkey)
    columns_list = [groupkey]
    columns_list.extend(res_col_name)
    df_res.columns = columns_list
    if asint:
        df_res[res_col_name] = df_res[res_col_name].astype(int)
    return df_res

# use example
# df_groupby(df,'appl_no', 'phone_gray_score', 'sum', 'phone_gray_score_sum', dup=False, asint=False)
# df_groupby(df,'appl_no', ['phone_gray_score'], ['sum'], ['phone_gray_score_sum'], dup=False, asint=False)
# df_groupby(df,'appl_no', ['register_cnt','phone_gray_score'], ['sum'], ['register_cnt_sum','phone_gray_score_sum'], dup=False, asint=False)
# df_groupby(df,'appl_no', ['register_cnt','phone_gray_score'], ['sum','avg','count'], ['register_cnt_sum','phone_gray_score_sum','register_cnt_avg','phone_gray_score_avg','register_cnt_count','phone_gray_score_count'], dup=False, asint=False)


# 1.2.1
def col_dummy(x, col, dummy_dict=[]):
    """
    function about:变量编码功能函数集
    by boysgs @20171103
    :param x: 一个数值
    :param col: df中需重新编码的变量名
    :param dummy_dict:  列表，变量所有取值组成，示例['value_1','value_2']
    :return:col_dummy_dict
    """
    dummy_dict_sorted = sorted(dummy_dict)
    dummy_dict_sorted_key = np.array(['_'.join(['if', col, i]) for i in dummy_dict_sorted])
    dummy_dict_sorted_value = [0] * len(dummy_dict_sorted_key)
    col_dummy_zip = zip(dummy_dict_sorted_key, dummy_dict_sorted_value)
    col_dummy_dict = dict((a, b) for a, b in col_dummy_zip)
    #
    if x in dummy_dict_sorted:
        col_dummy_dict['_'.join(['if', col, x])] = 1
    return col_dummy_dict

# use example
# df = pd.DataFrame({'col1': [1, np.nan, 2, 3], 'col2': [3, 4, 5, 1], 'col3': ['s', 'a', 'c', 'd']})
# dummy_dict = ['a', 'b', 'c', 'd', 's']
# col = 'col3'
# DataFrame(list(df[col].apply(lambda x: col_dummy(x, col, dummy_dict))))


# 1.2.2
def col_dummy_lb(x, lb_trans, sorted_dummy_varname_list=[]):
    """
    function about:变量编码功能函数集（使用LabelBinarizer方法）
    by boysgs @20171103
    :param x: 一个数值
    :param lb_trans: 一个变量利用preprocessing.LabelBinarizer 方法生成的对象
    :param sorted_dummy_varname_list: 列表，升序排列的变量所有取值组成，示例['value_1','value_2']
    :return:col_dummy_dict 字典
    """
    dummy_value = lb_trans.transform(str([x]))
    col_dummy_dict = dict(zip(sorted_dummy_varname_list, dummy_value[0]))
    return col_dummy_dict


# 2.1
def meetOneCondition(x,symbol = '=',threshold = ('None','b')):
    """
    # 输入:
    # 变量名：年龄
    # 符号：=，！=，>，< , >=, <= , in , not in，like, not like
    # 阈值：10,(10,11),'%10%'
    # 输出
    # 满足条件输出1，否则输出0
    """

    if pd.isnull(x) or x == '':
        if symbol in ['!=','not in ','not like'] and threshold!='None':
            return 1
        elif threshold=='None':
            if symbol == '=':
                return 1
            elif symbol == '!=':
                return 0
        else:
            return 0
    elif symbol == '=':
        if threshold=='None':
            return 0
        elif x == threshold:
            return 1
        else:
            return 0
    elif symbol == '!=':
        if threshold=='None':
            return 1
        elif x != threshold:
            return 1
        else:
            return 0
    elif symbol == '>':
        if x > threshold:
            return 1
        else:
            return 0
    elif symbol == '<':
        if x < threshold:
            return 1
        else:
            return 0
    elif symbol == '>=':
        if x >= threshold:
            return 1
        else:
            return 0
    elif symbol == '<=':
        if x <= threshold:
            return 1
        else:
            return 0
    elif symbol == 'in':
        if x in threshold:
            return 1
        else:
            return 0
    elif symbol == 'not in':
        if x not in threshold:
            return 1
        else:
            return 0
    elif symbol == 'like':
        if threshold[0] == '%' and threshold[-1] == '%':
            if threshold[1:-1] in x:
                return 1
            else:
                return 0
        if threshold[0] == '%' and threshold[-1] != '%':
            if threshold[1:] == x[len(x)-len(threshold[1:]):]:
                return 1
            else:
                return 0
        if threshold[0] != '%' and threshold[-1] == '%':
            if threshold[0:-1] == x[0:len(threshold[0:-1])]:
                return 1
            else:
                return 0
        else:
            return 'you need cheack your "like" threshold'
    elif symbol == 'not like':
        if threshold[0] == '%' and threshold[-1] == '%':
            if threshold[1:-1] not in x:
                return 1
            else:
                return 0
        if threshold[0] == '%' and threshold[-1] != '%':
            if threshold[1:] != x[len(x)-len(threshold[1:]):]:
                return 1
            else:
                return 0
        if threshold[0] != '%' and threshold[-1] == '%':
            if threshold[0:-1] != x[0:len(threshold[0:-1])]:
                return 1
            else:
                return 0
        else:
            return 'you need cheack your "not like" threshold'
    elif symbol =='regex':
        if re.search(threshold,x):
            return 1
        else:
            return 0
    else:
        return 'please contact the developer for increaing then type of the symbol'

# test:
# x = 'abcde'
# meetOneCondition(x,'=','abcd2')
# meetOneCondition(x,'like','abc%')
# meetOneCondition(x,'like','%abc')
# meetOneCondition(x,'regex','b|adz|z')


# 2.2
def meetMultiCondition(condition = ((),'and',())):
    """
    # 输入
    # 多个条件，单个条件参考meetOneCondition中的
    # 例子 condition = ( ('age','>=',18), 'and', ( ('age','<=',40),'or',('gender','=','female') ) )
    # 输出
    # 满足条件输出1，否则输出0
    """
    if 'and' in condition:
        a = [k for k in condition if k!='and']
        b = []
        for l in range(len(a)):
            b.append(meetMultiCondition(a[l]))
        if 0 in b:
            return 0
        else:
            return 1
    if 'or' in condition:
        a = [k for k in condition if k != 'or']
        b = []
        for l in range(len(a)):
            b.append(meetMultiCondition(a[l]))
        if 1 in b:
            return 1
        else:
            return 0
    else:
        return meetOneCondition(condition[0],condition[1],condition[2])

# test
# zz ='abcde'
# yy = 10
# xx = 5
# meetMultiCondition(((zz,'=','abc'),'or',(yy,'>',7)))


# 2.3
def singleConditionalAssignment(conditon =('z','=',('None','b')),assig1=1, assig2=0):
    """
    # 单条件赋值
    # 输入
    # 参考meetOneCondition的输入
    # 例如：conditon = ('age','>=',18)
    # 输出：
    # 满足条件assig1
    # 不满足条件assig2
    """
    if meetOneCondition(conditon[0],conditon[1],conditon[2])==1:
        return assig1
    elif meetOneCondition(conditon[0], conditon[1], conditon[2]) == 0:
        return assig2
    else:
        return meetOneCondition(conditon[0],conditon[1],conditon[2])

# test
# singleConditionalAssignment((x, '=', 'abcde'), 5, 1)


# 2.4
def multiConditionalAssignment(condition = (),assig1 = 1,assig2 = 0):
    """
    # 多个条件赋值
    ###输入
    ##多个条件类似meetMultiCondition的输入
    ###输出：
    ##满足条件assig1
    ##不满足条件assig2
    """
    if meetMultiCondition(condition)==1:
        return assig1
    else:
        return assig2

# test
# xx=5
# multiConditionalAssignment(condition =((zz,'=','abcde'),'and',( (yy,'>',10), 'or', (xx,'=',5) )),assig1 = 999,assig2 = 0)


# 2.5
def multiConditionalMultAssignment(condition = ((('zz','not in', ('硕士','博士')),1),(('zz','not in', ('硕士','博士')),2)),assig = 0):
    """
    ####多个条件多个赋值
    ###输入
    ##多个条件类似meetMultiCondition的输入,再加一满足的取值

    ###输出：
    ##满足条件输出输入目标值
    ##不满足条件assig
    """
    for l in condition:
        if meetMultiCondition(l[0])==1:
            return l[1]
    return assig

# test
# multiConditionalMultAssignment((((zz,'=','abcdef'),1),((zz,'=','abcde'),2)),3)


# 3.1
def substring(string,length,pos_start=0):
    """
    function about : 字符串截取
    by dabao @20171106
    :param string:  被截取字段
    :param length: 截取长度
    :param pos_start: 从第几位开始截取,defualt=0
    :return:  a string :substr 
    """
    pos_end = length + pos_start
    if string is np.NaN:
        return np.NaN
    else:
        str_type = type(string)
        if str_type==str:
            substr = string[pos_start:pos_end]
        else:
            string = str(string)
            substr = string[pos_start:pos_end]        
        return substr

# test
# string=370321199103050629
# length=4
# pos_start=6
# substring(string,length,pos_start)
# string=np.NaN


# 3.2
def charindex(substr,string,pos_start=0):
    """
    function about : 字符串位置查询
    by dabao @20171106
    :param substr
    :param string: substr 在 string 起始位置
    :param pos_start: 查找substr的开始位置,default=0
    :return:  a int :substr_index 
    """
    if string is np.NaN:
        return np.NaN
    else:
        substr = str(substr)
        string = str(string)
        substr_index = string.find(substr,pos_start)
        return substr_index

# test
# string='370321199103050629'
# substr='1991'
# charindex(substr,string)
# string.find(substr,0)


# 3.3
def trim(string,substr=' ',method='both'):
    """
    function about : 删除空格或其他指定字符串
    by dabao @20171106
    :param string: a string
    :param substr: 在string两端删除的指定字符串,default=' '
    :param method: 删除方式:left 删除左边, right 删除右边, both 删除两边
    :return:  a string :string_alter 
    """
    if string is np.NaN:
        return np.NaN
    else:        
        substr = str(substr)
        string = str(string)
        if method in ['left','right','both']:               
            if method =='left':
                string_alter = string.lstrip(substr)
            elif method == 'right':
                string_alter = string.rstrip(substr)
            elif method == 'both':
                string_alter = string.strip(substr)
        else:
            string_alter = string.strip(substr)
            print("Warning: method must be in ['left','right','both']! If not, the function will be acting as 'both'")    
        return string_alter

# test:
# string='  OPPO,HUAWEI,VIVO,HUAWEI '
# trim(string)
# （4）计算字符串长度：SQL中的LEN（）函数 ,python自带 len()
# （5）字符串转换为大、小写：SQL 中的 LOWCASE,UPPER 语句,python自带函数 string.upper(),string.lower()


# 3.4
def OnlyCharNum(s,oth=''):
    # 只显示字母与数字
    s2 = s.lower()
    fomart = 'abcdefghijklmnopqrstuvwxyz0123456789'
    for c in s2:
        if not c in fomart:
            s = s.replace(c,'')
    return s


# 4.1
def dateformat(date,symbol):
    """
    输入:
     变量名：时间,按照格式接收10位、19位
     可选：'year','month','day','hour','minute','second'
    输出
     满足条件输出值，否则报错
    """
    if pd.isnull(date):
        return np.NaN
    date = str(date)
    if len(date)==10:
        date=date+' 00:00:00'
    date=dt.datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
    if symbol in ['year','month','day','hour','minute','second']:               
        if symbol =='year':
            datetime_elect = date.year
        elif symbol == 'month':
            datetime_elect = date.month
        elif symbol == 'day':
            datetime_elect = date.day
        elif symbol == 'hour':
            datetime_elect = date.hour
        elif symbol == 'minute':
            datetime_elect = date.minute
        elif symbol == 'second':
            datetime_elect = date.second
    else:
        datetime_elect = np.NaN
        print("Warning: symbol must be in ['year','month','day','hour','minute','second']! If not, the function will be acting as 'both'")    
    return datetime_elect
        
# test1:
# dateformat('2017-09-25 12:58:45','day')
# dateformat('2017-09-25 12:58:45','hour')
# dateformat('2017-09-25','day')
# dateformat(null,'hour')


# 4.2
def datediff(symbol,date_begin,date_end):
    """
    输入:
     变量名：时间,按照格式接收10位、19位
     可选：'year','month','day','hour','minute','second'
    输出
     满足条件输出值，否则报错
    """
    if pd.isnull(date_begin) or pd.isnull(date_end):
        return np.NaN
    date_begin = str(date_begin)
    date_end = str(date_end)
    if len(date_begin)==4:
        date_begin=date_begin+'-01-01 00:00:00'
    if len(date_end)==4:
        date_end=date_end+'-01-01 00:00:00'
    if len(date_begin)==7:
        date_begin=date_begin+'-01 00:00:00'
    if len(date_end)==7:
        date_end=date_end+'-01 00:00:00'
    if len(date_begin)==10:
        date_begin=date_begin+' 00:00:00'
    if len(date_end)==10:
        date_end=date_end+' 00:00:00'
    date_begin=dt.datetime.strptime(date_begin,'%Y-%m-%d %H:%M:%S')
    date_end=dt.datetime.strptime(date_end,'%Y-%m-%d %H:%M:%S')
    if symbol in ['year','month','day','hour','minute','second']:
        r =  relativedelta.relativedelta(date_end,date_begin)             
        if symbol =='year':
            datetime_diff=r.years
        elif symbol == 'month':
            datetime_diff=r.years*12+r.months
        elif symbol == 'day':
            datetime_diff = (date_end-date_begin).days
        elif symbol == 'hour':
            datetime_days = (date_end-date_begin).days
            datetime_seconds = (date_end-date_begin).seconds
            datetime_diff = datetime_seconds/3600+datetime_days*24
        elif symbol == 'minute':
            datetime_days = (date_end-date_begin).days
            datetime_seconds = (date_end-date_begin).seconds
            datetime_diff=datetime_seconds/60+datetime_days*24*60
        elif symbol == 'second':
            datetime_days = (date_end-date_begin).days
            datetime_seconds = (date_end-date_begin).seconds
            datetime_diff=datetime_seconds+datetime_days*24*60*60
    else:
        datetime_diff = np.NaN
        print("Warning: symbol must be in ['year','month','day','hour','minute','second']! If not, the function will be acting as 'both'")    
    return datetime_diff

# test
# datediff('month','2013','2017-09-25 12:58:45')
# datediff('day','2017-09-25','2017-12-30')
# datediff('hour','2017-09-15 10:58:45','2017-09-25 12:58:45')
# datediff('day','2017-09-25','2017-12-30 12:58:45')