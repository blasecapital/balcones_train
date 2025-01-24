# decorators.py


# Define the decorator first
def public_method(func):
    func.is_public = True
    return func
