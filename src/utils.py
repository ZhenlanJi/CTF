from datetime import datetime
import functools

FUNCTION_TIME_DICT = {}

DATA_COLUMNS = ["Data_Cons", "Data_sens_DP", "Data_sens_SPD",
                "Data_other_DP", "Data_other_SPD"]
TRAIN_SENS_COLUMNS = ["Train_sens_DP", "Train_sens_SPD", "Train_sens_AOD", ]
TRAIN_OTHER_COLUMNS = ["Train_other_DP", "Train_other_SPD", "Train_other_AOD", ]
TEST_SENS_COLUMNS = ["Test_sens_DP", "Test_sens_SPD", "Test_sens_AOD"]
TEST_OTHER_COLUMNS = ["Test_other_DP", "Test_other_SPD", "Test_other_AOD"]
INDIVIDUAL_COLUMNS = ["Train_Cons", "Train_TI", "Train_CDS",
                      "Test_Cons", "Test_TI", "Test_CDS"]
PERF_COLUMNS = ["Train_Acc", "Train_Loss", "Train_F1", "Test_Acc", "Test_Loss", "Test_F1"]
OTHER_COLUMNS = ["AE_FGSM", "AE_PGD", "MI_BlackBox", "Ratio", "Model_Width"]

METHOD_COLUMNS = [
    # Pre-processing
    "FairMask", "Fairway", "FairSmote", "LTDD", "DIR", "RW", 
    # In-processing
    "AdDebias", "EGR", "PR", 
    # Post-processing
    "EO", "CEO", "ROC"]

ALL_COLUMNS = DATA_COLUMNS + TRAIN_SENS_COLUMNS + TRAIN_OTHER_COLUMNS + TEST_SENS_COLUMNS + \
    TEST_OTHER_COLUMNS + INDIVIDUAL_COLUMNS + PERF_COLUMNS + OTHER_COLUMNS


def time_statistic(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        cost = end_time - start_time
        if func.__name__ not in FUNCTION_TIME_DICT:
            FUNCTION_TIME_DICT[func.__name__] = (cost, 1)
        else:
            FUNCTION_TIME_DICT[func.__name__] = (
                FUNCTION_TIME_DICT[func.__name__][0] + cost, FUNCTION_TIME_DICT[func.__name__][1] + 1)
        # print("Time cost of", func.__name__, "is", end_time - start_time)
        return result
    return wrapper


def logme(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Calling function", func.__name__)
        return func(*args, **kwargs)
    return wrapper


def print_time_statistic():
    for func_name, (cost, count) in FUNCTION_TIME_DICT.items():
        print(f"Function: {func_name}, time cost: {cost}, call count: {count}")
