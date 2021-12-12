import string
import random
from typing import Optional
from os import getcwd

import uvicorn
from fastapi.responses import FileResponse
from fastapi import FastAPI, Body
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI
from pydantic import BaseModel
import os
import json

from starlette.responses import RedirectResponse

from processing_mirror import generate_regex


def rnd_str(n: int = 15):
	return (
			'file_'
			+ ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))
			+ '.txt'
	)


def write_file(filename: str, data: str):
	with open(os.path.join('sv_data/req/', filename), 'w+') as f:
		f.write(data)


class InputLog(BaseModel):
	input: str


def general_processing(s: str):
	uri_s = []
	try:
		s = s.strip().split('\n')
		for line in s:
			print('line', line)
			structure = json.loads(line)
			uri_s.append(structure["request"]["request_uri"])
		print(uri_s)
		return uri_s
	except Exception:
		print('Not OK')
		return None
	

app = FastAPI(port=8000)
app.mount("/static", StaticFiles(directory="JunoX"), name="static")


@app.get("/")
def read_root():
	response = RedirectResponse(url='/static/index.html')
	return response


@app.get("/pings")
def read_item():
	return {"status": "OK"}


# submit logs
@app.post("/api/Regexs")
def uploadData(input: str = Body(..., embed=True)):
	filename = rnd_str()
	uri_s = general_processing(input)
	
	if uri_s:
		write_file(filename, ', '.join(uri_s))
		# rg_dict = generate_regex(filename)
		# print(rg_dict.values())
		return {'filename': filename}
	else:
		return {'status': '?'}


if __name__ == '__main__':
	uvicorn.run(app, port=8000, host='localhost')
