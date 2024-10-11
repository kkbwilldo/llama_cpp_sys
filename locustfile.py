from locust import HttpUser, TaskSet, task, constant

class LlamaCPPTest(TaskSet):
    """
    llama.cpp 서버에 요청을 보내는 테스트 클래스
    """
    @task
    def llama_request(self):
        """
        llama.cpp 서버에 요청을 보내는 함수
        """
        # llama-server로 보낼 요청 데이터 예시
        payload = {
            "prompt": "Building a website can be done in 10 simple steps:",
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
    wait_time = constant(10) # 사용자 간 요청 간격
