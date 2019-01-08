def cheak_type(_input, expect_type, message=None):
    if not isinstance(_input, expect_type):
        if message is not None:
            raise TypeError(str(message))
        elif type(expect_type) is tuple:
            types_str = "("
            for t in expect_type:
                types_str += t.__name__ + ", "

            types_str += ")"
            raise TypeError("expect one of %s, got %s" %(types_str, type(_input).__name__))
        else:
            raise TypeError("expect %s, got %s" %(expect_type.__name__, type(_input).__name__))

def cheak_interval(_input, mini = -float("inf"), maxi = float("inf"),
                   left_close=False, right_close=False, message=None):
    # cheak mini<(=)input<(=)maxi
    try:
        _input < float("Inf")
    except:
        raise TypeError("expect something in extanded realnumber, got %s" %(type(_input).__name__))
        # extanded realnumber because infinite is OK
    
    left_cheak = (_input > mini) or (left_close and _input == mini)
    right_cheak = (_input < maxi) or (right_close and _input == maxi)
    if not(left_cheak and right_cheak):
        if message is None:
            if left_close:
                left_compare = "<="
            else:
                left_compare = "<"

            if right_close:
                right_compare = "<="
            else:
                right_compare = "<"

            raise ValueError("expect %s %s value %s %s, got %s" %(mini, left_compare,right_compare, maxi, _input))
        else:
            raise ValueError(str(message))

def cheak_range(_input, mini = -float("inf"), maxi = float("inf"), message=None):
    # cheak inupt is int and mini<=input<=maxi
    if not isinstance(_input, int):
        raise TypeError("expect int, got %s" %(type(_input).__name__))
    
    left_cheak = (_input >= mini)
    right_cheak = (_input <= maxi)
    if not(left_cheak and right_cheak):
        if message is None:
            raise ValueError("expect int and %s <= value <= %s, got %s" %(mini, maxi, _input))
        else:
            raise ValueError(str(message))