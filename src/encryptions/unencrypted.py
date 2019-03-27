
'''
# this function is used for training
def encrypt(inputs):
    return (inputs / 255.0) - 0.5
'''


# this function is used when running CW_attack
def encrypt(inputs):
    return inputs


def print_encryption_details(out):
    pass
