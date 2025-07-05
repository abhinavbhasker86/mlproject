import sys
import logging

#Function to generate a detailed error message including file name and line number
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() # Get traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename # Get the name of the file where the error occurred

    # Construct the error message with file name, line number, and error message
    error_message=" Error Occured in python script name: [{0}], line Number: [{1}], error message: [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message

# Custom exception class to handle exceptions with detailed error messages
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):  
        super().__init__(error_message) # Call the base class constructor with the error message
        # Store the detailed error message
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        # Return the detailed error message when the exception is printed
        return self.error_message


    