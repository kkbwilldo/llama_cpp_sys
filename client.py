import asyncio
import aiohttp
import time
from typing import Dict, List

def read_prompt_from_file(file_path: str) -> str:
    """
    지정된 파일에서 프롬프트를 읽어옵니다.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


async def send_request(session: aiohttp.ClientSession, url: str, payload: Dict) -> float:
    """
    비동기적으로 단일 HTTP 요청을 보내고 처리 시간을 측정합니다.

    Args:
        session: aiohttp.ClientSession 객체
        url: 요청을 보낼 URL
        payload: 요청에 포함할 JSON 페이로드

    Returns:
        요청 처리 시간 (초)
    """
    start_time = time.time()
    async with session.post(url, json=payload) as response:
        result = await response.text()
    return time.time() - start_time, result


async def run_benchmark(url: str, payload: Dict, num_requests: int) -> (float, List[float]):
    """
    지정된 수의 동시 요청을 비동기적으로 실행하고 전체 소요 시간 및 각 요청의 처리 시간을 측정합니다.

    Args:
        url: 요청을 보낼 URL
        payload: 요청에 포함할 JSON 페이로드
        num_requests: 동시 요청 수

    Returns:
        전체 소요 시간 (초) 및 각 요청의 처리 시간 목록
    """
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        tasks = [send_request(session, url, payload) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        individual_times, responses = zip(*results)
        return total_time, individual_times, responses


def main():
    """
    메인 함수: 다양한 동시 요청 수에 대해 벤치마크를 실행하고 결과를 출력합니다.
    """
    url = "http://localhost:8080/completion"  # llama.cpp 서버 주소
    prompt = read_prompt_from_file("./outputs/harry_potter.txt")
    payload = {
        "prompt": prompt,
        "n_predict": 768,
    }

    num_requests_list = [1, 2, 4, 8]
    # num_requests_list = [16]
    
    for num_requests in num_requests_list:
        print(f"Running benchmark with {num_requests} simultaneous requests...")
        total_time, individual_times, responses = asyncio.run(run_benchmark(url, payload, num_requests))
        
        avg_time = sum(individual_times) / len(individual_times)
        print(f"Total time for {num_requests} requests: {total_time:.4f} seconds")
        print(f"Average time per request: {avg_time:.4f} seconds")
        print(f"Throughput: {num_requests / total_time:.2f} requests/second")
        print()

if __name__ == "__main__":
    main()
