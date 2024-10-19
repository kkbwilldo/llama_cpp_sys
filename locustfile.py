from locust import HttpUser, TaskSet, task, constant

class LlamaCPPTest(TaskSet):
    """
    llama.cpp 서버에 요청을 보내는 테스트 클래스
    """
    def on_start(self):
        """
        TaskSet이 시작될 때 실행되는 함수
        harry_potter.txt 파일을 읽어 lines 리스트에 저장
        """
        with open("./outputs/harry_potter.txt", "r") as file:
            self.prompt = file.read().strip()


    @task
    def llama_request(self):
        """
        llama.cpp 서버에 요청을 보내는 함수
        """
        # llama-server로 보낼 요청 데이터
        payload = {
            "prompt": self.prompt,
            "n_predict": 768
        }

        # llama-server에 /completion 경로로 POST 요청 보내기
        self.client.post("/completion", json=payload)

class LlamaUser(HttpUser):
    """
    llama.cpp 서버에 요청을 보내는 사용자 클래스
    """
    tasks = [LlamaCPPTest]
    host = "http://localhost:8080"  # llama-server 주소
    wait_time = constant(20) # 사용자 간 요청 간격
