FROM public.ecr.aws/lambda/python:3.11

# Copy Lambda code
COPY lambda_script.py ${LAMBDA_TASK_ROOT}/
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Upgrade pip & install dependencies
RUN pip install --upgrade pip
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

CMD ["lambda_script.lambda_handler"]
