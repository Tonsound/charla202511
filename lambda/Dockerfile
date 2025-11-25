FROM public.ecr.aws/lambda/python:3.11

# Copy function code
COPY lambda_script.py ${LAMBDA_TASK_ROOT}/
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install the specified packages
RUN pip install -r requirements.txt

# Command can be your handler
CMD ["lambda_script.lambda_handler"]
