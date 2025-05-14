import inspect

def my_function(arg1, arg2, kwarg1=None):
    pass

# Get the function signature
num_args = len(inspect.signature(my_function).parameters)


# Print the number of arguments (both positional and keyword)
print(f"The function has {num_args} parameters.")
