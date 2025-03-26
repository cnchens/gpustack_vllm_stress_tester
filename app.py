import threading
import time
from openai import OpenAI
from tqdm import tqdm

# gpustack configurations
gpustack_server_ip = "localhost"
gpustack_server_port = "11434" # Use 80 or 443 in gpustack api
gpustack_api_use_https = False
gpustack_api_path = "/v1" # Instead use "v1-openai" in gpustack api
gpustack_api_key = "ollama"

# global model configurations
model = "deepseek-r1:8b"
model_temperature = 1
model_max_tokens = 1024
model_top_p = 1
model_messages = [
    {"role" : "system", "content" : "写一篇宫傲大战老奶奶的故事，1000字"}
]
model_stream_output = True # Enable stream output, always left True

# test configurations
test_threads = 1

# other configurations
screen_flush = 0.1

def configure_api():
    if gpustack_api_use_https == True:
        use_https = "https"
    else:
        use_https = "http"

    client = OpenAI(
        base_url = f"{use_https}://{gpustack_server_ip}:{gpustack_server_port}{gpustack_api_path}", 
        api_key = gpustack_api_key
    )

    return client

def api_start_chat(model, temperature, max_tokens, top_p, messages, stream_output):
    client = configure_api()

    response = client.chat.completions.create(
        model = model, 
        temperature = temperature, 
        max_tokens = max_tokens, 
        top_p = top_p, 
        messages = messages, 
        stream = stream_output
    )

    return response

def api_stream_get_response(model, temperature, max_tokens, top_p, messages, stream_output):
    # output_chunk_buffer = []
    output_message_buffer = []
    output_word_count = 0

    response = api_start_chat(model, temperature, max_tokens, top_p, messages, stream_output)
    time_start = time.time()

    with tqdm(total=model_max_tokens, desc="Thread 01") as pbar:
        for i in response:
            # output_chunk_buffer.append(i)
            chunk_message = i.choices[0].delta.content
            output_message_buffer.append(chunk_message)

            pbar.update(1)

            output_word_count = output_word_count + 1

            # print(chunk_message, end="")

    time_stop = time.time()

    return output_message_buffer, output_word_count, time_start, time_stop
    

output_message_buffer, output_word_count, time_start, time_stop = api_stream_get_response(model, model_temperature, model_max_tokens, model_top_p, model_messages, model_stream_output)
time.sleep(1)
print(f"总共输出了 {output_word_count}/{model_max_tokens} 字")
print(f"用时 {(time_stop - time_start)} 秒")
print(f"每秒输出 {output_word_count / (time_stop - time_start)} 字")
time.sleep(3)
print("\n原始数据\n")
for i in output_message_buffer:
    print(i, end="")

