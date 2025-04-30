from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import time
import os
import aiofiles
import json
import csv
import re
from src.helper import llm_pipeline  # Make sure this module exists and is importable

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = "static/docs/"
    os.makedirs(base_folder, exist_ok=True)  # Create directory if it doesn't exist

    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, "wb") as f:
        await f.write(pdf_file)

    response_data = json.dumps({"msg": "success", "pdf_filename": pdf_filename})
    return Response(content=response_data, media_type="application/json")


def clean_text(text):
    # Remove markdown formatting like **Q:** and **A:**
    text = re.sub(r'\*\*([QA]):\*\*', '', text)
    # Remove numbered prefixes like "1. "
    text = re.sub(r'^\d+\.\s*', '', text)
    # Remove any other formatting markers
    text = text.strip()
    return text


def get_csv(filepath):
    try:
        answer_generation_chain, ques_list = llm_pipeline(filepath)
        ques_list=ques_list[1:]
        base_folder = 'static/output/'
        os.makedirs(base_folder, exist_ok=True)

        output_file = os.path.join(base_folder, "QA.csv")

        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Question', 'Answer'])

            for ques in ques_list:
                # Clean the question text
                cleaned_question = clean_text(ques)
                # print("Question: ", cleaned_question)

                answer_result = answer_generation_chain.invoke(cleaned_question)

                if isinstance(answer_result, dict):
                    if 'result' in answer_result:
                        answer = answer_result['result']
                    else:
                        # Use str() as fallback if we can't extract a specific field
                        answer = str(answer_result)
                else:
                    answer = answer_result

                # Clean the answer text
                cleaned_answer = clean_text(answer)
                print("Answer: ", cleaned_answer)
                time.sleep(2)
                print("----------------------------------------\n\n")

                csv_writer.writerow([cleaned_question, cleaned_answer])

        return output_file
    except Exception as e:
        print(f"Error generating CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/analyze")
async def analyze(request: Request, pdf_filename: str = Form(...)):
    try:
        output_file = get_csv(pdf_filename)
        return JSONResponse(content={"output_file": output_file})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8090, reload=True)