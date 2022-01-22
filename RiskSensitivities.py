

def computeFirstDerivative(v_u_plus_du , v_u_minus_du , du):
    '''
    Compute the first derivatve of a function using central difference
    @var v_u_plus_du: is the value of the function computed for a positive bump amount du
    @var v_u_minus_du: is the value of the function computed for a negative bump amount du
    @var du: bump amount
    '''
    first_derivative = (v_u_plus_du - v_u_minus_du) / (2.0 * du)
    return first_derivative

def computeSecondDerivative(v_u, v_u_plus_du, v_u_minus_du, du):
    '''
    Compute the second derivatve of a function using central difference
    @var v_u: is the value of the function
    @var v_u_plus_du: is the value of the function
    computed for a positive bump amount du
    @var v_u_minus_du: is the value of the function
    computed for a negative bump amount du
    @var du: bump amount
    '''
    second_derivative = ((v_u_plus_du - 2.0*v_u + v_u_minus_du)/(du * du))
    return second_derivative