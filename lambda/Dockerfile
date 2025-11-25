FROM public.ecr.aws/lambda/python:3.11

# Install build essentials for compiling packages
RUN yum install -y gcc gcc-c++ make && yum clean all

# Copy your code
COPY lambda_script.py ${LAMBDA_TASK_ROOT}/
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

CMD ["lambda_script.lambda_handler"]
