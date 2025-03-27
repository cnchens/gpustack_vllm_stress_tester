import threading
import time
from openai import OpenAI
from tqdm import tqdm

# gpustack configurations
gpustack_server_ip = "10.201.1.1"
gpustack_server_port = "80" # Use 80 or 443 in gpustack api
gpustack_api_use_https = False
gpustack_api_path = "/v1-openai" # Instead use "v1-openai" in gpustack api
gpustack_api_key = "gpustack_7ae5cca85e0f708e_9beeb02968baf0ff817a250e1fa5ab8b"

# global model configurations
model = "deepseek-r1:7b_fp16_ty"
model_temperature = 1
model_max_tokens = 256
model_top_p = 1
model_messages = [
    {"role" : "system", "content" : "写一篇1000字左右的论文，题目不限"}
]
model_stream_output = True # Enable stream output, always left True

# threading configurations
run_threads = 20
run_threads_f = []
# thread_is_running = 0

# other configurations
# screen_flush = 0.1

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

def api_stream_get_response(model, temperature, max_tokens, top_p, messages, stream_output, thread_name):
    # output_chunk_buffer = []
    output_message_buffer = []
    output_word_count = 0

    response = api_start_chat(model, temperature, max_tokens, top_p, messages, stream_output)
    time_start = time.time()

    with tqdm(total=model_max_tokens, desc=thread_name) as pbar:
        for i in response:
            # output_chunk_buffer.append(i)
            # chunk_message = i.choices[0].delta.content
            # output_message_buffer.append(chunk_message)

            pbar.update(1)

            output_word_count = output_word_count + 1

            # print(chunk_message, end="")

    time_stop = time.time()

    return output_message_buffer, output_word_count, time_start, time_stop

class runThread(threading.Thread):
    def __init__(self, thread_id, thread_name, thread_counter):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.thread_name = thread_name
        self.thread_counter = thread_counter

    def run(self):
        print("启动线程" + self.name)
        output_message_buffer, output_word_count, time_start, time_stop = api_stream_get_response(model, model_temperature, model_max_tokens, model_top_p, model_messages, model_stream_output, self.name)
        print(f"-------------------- {self.name} report start --------------------\n{self.name}: 总共输出了 {output_word_count}/{model_max_tokens} 字\n{self.name}: 用时 {(time_stop - time_start)} 秒\n{self.name}: 每秒输出 {output_word_count / (time_stop - time_start)} 字\n-------------------- {self.name} report stop --------------------\n")
        print("退出线程" + self.name)

def app():
    for i in range(run_threads):
        thread_n = runThread(i, f"Thread-{i}", i)
        thread_n.start()
        run_threads_f.append(thread_n)

if __name__ == "__main__":
    time_all_start = time.time()

    app()

    for j in run_threads_f:
        j.join()

    time_all_stop = time.time()

    # print(f"输出{run_threads}线程总共用时{time_all_stop - time_all_start}秒\n平均每个线程{(time_all_stop - time_all_start) / run_threads}秒")
