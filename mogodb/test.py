import requests
from pymongo import MongoClient
import math
import json
#### 페이징을 통해 totalCnt만큼 데이터적재 ####

# 1. API 정보 및 MongoDB 연결 정보 설정
OC = "balet99c"  # 실제 발급받은 인증키를 사용해야 합니다.
TARGET_LAW = "law"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "law_data"
COLLECTION_NAME = "laws3"

def get_all_law_ids():
    """
    lawSearch.do API를 호출하여 모든 법령 ID 목록을 가져옵니다.
    - 전체 데이터 수를 기반으로 페이징 처리를 통해 모든 ID를 수집합니다.
    """
    print("모든 법령 ID 목록 가져오기를 시작합니다.")
    law_ids = []
    page = 1
    num_of_rows = 100  # 한 번에 100개씩 요청

    # 첫 페이지를 호출하여 전체 데이터 수(totalCnt)를 확인
    try:
        list_url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={OC}&target={TARGET_LAW}&type=JSON&display={num_of_rows}&page={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }
        
        response = requests.get(list_url, headers=headers)
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        
        data = response.json()
        
        # --- [수정된 부분 시작] ---
        
        # LawSearch 키가 있는지, 그리고 그 안에 데이터가 있는지 확인
        if "LawSearch" not in data or not data["LawSearch"]:
            print("API 응답에서 'LawSearch' 데이터를 찾을 수 없거나 데이터가 비어있습니다.")
            # API가 에러 메시지를 반환하는 경우를 대비하여 추가
            if "faultInfo" in data:
                print(f"API 오류: {data.get('faultInfo', {}).get('message')}")
            return []

        law_search_data = data["LawSearch"]

        total_count = int(law_search_data.get("totalCnt", 0))
        if total_count == 0:
            print("조회할 법령이 없습니다.")
            return []

        print(f"총 {total_count}개의 법령을 확인했습니다. 페이지별로 ID를 수집합니다.")
        
        # 첫 페이지의 ID 추가
        law_list = law_search_data.get("law", [])
        for law in law_list:
            if law.get("법령ID"):
                law_ids.append(law.get("법령ID"))

        # 나머지 페이지에 대해 반복 호출
        total_pages = math.ceil(total_count / num_of_rows)
        for page_num in range(2, total_pages + 1):
            list_url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={OC}&target={TARGET_LAW}&type=JSON&display={num_of_rows}&page={page_num}"
            # 모든 요청에 헤더를 포함하는 것이 좋습니다.
            response = requests.get(list_url, headers=headers) 
            response.raise_for_status()
            
            page_data = response.json()
            
            if "LawSearch" in page_data:
                law_list = page_data["LawSearch"].get("law", [])
                for law in law_list:
                    if law.get("법령ID"):
                        law_ids.append(law.get("법령ID"))
            
            print(f"{page_num}/{total_pages} 페이지 처리 완료...")

        # --- [수정된 부분 끝] ---

    except requests.exceptions.RequestException as e:
        print(f"법령 목록 API 호출 중 오류 발생: {e}")
        return []
    # response.json() 에서 JSON 변환 실패 시 발생
    except json.JSONDecodeError:
        print("법령 목록 API 응답이 JSON 형식이 아닙니다.")
        print("받은 응답 내용:", response.text)
        return []

    print(f"총 {len(law_ids)}개의 법령 ID를 성공적으로 가져왔습니다.")
    return law_ids

def fetch_and_store_law_data(law_id, collection):
    """
    특정 법령 ID의 상세 정보를 가져와 MongoDB에 저장합니다.
    """
    api_url = f"https://www.law.go.kr/DRF/lawService.do?OC={OC}&target={TARGET_LAW}&type=json&ID={law_id}"
    
    try:
        print(f"법령ID {law_id} 데이터 가져오는 중...")
        response = requests.get(api_url)
        response.raise_for_status()
        law_data = response.json()

        # 법령ID를 기준으로 이미 데이터가 있는지 확인하고, 없으면 새로 추가 (upsert)
        query = {"법령ID": law_data.get("법령ID", law_id)}
        collection.update_one(query, {"$set": law_data}, upsert=True)
        print(f"법령ID {law_id} 데이터를 성공적으로 저장/업데이트했습니다.")

    except requests.exceptions.RequestException as e:
        print(f"API 호출 중 오류가 발생했습니다 (ID: {law_id}): {e}")
    except ValueError:
        print(f"JSON 데이터 파싱 중 오류가 발생했습니다 (ID: {law_id}).")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다 (ID: {law_id}): {e}")

if __name__ == "__main__":
    # 1. 모든 법령 ID 가져오기
    all_law_ids = get_all_law_ids()

    if not all_law_ids:
        print("작업을 중단합니다.")
    else:
        client = None
        try:
            # 2. MongoDB 연결
            client = MongoClient(MONGO_URI)
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            print(f"'{MONGO_URI}'의 '{DB_NAME}' 데이터베이스에 연결했습니다.")

            # 3. 각 법령 ID에 대해 데이터 가져오기 및 저장
            for law_id in all_law_ids:
                fetch_and_store_law_data(law_id, collection)
            
            print("모든 법령 데이터 처리를 완료했습니다.")

        except Exception as e:
            print(f"MongoDB 처리 중 오류 발생: {e}")
        finally:
            # 4. MongoDB 연결 종료
            if client:
                client.close()
                print("MongoDB 연결을 종료했습니다.")
