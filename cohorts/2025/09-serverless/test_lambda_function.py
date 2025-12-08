import lambda_function

# only works if classifier_path in lambda_function.py exists as model
# the problem is that docker and local model have different names

event = {
    "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
}

print(lambda_function.lambda_handler(event, None))
