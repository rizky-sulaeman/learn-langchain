from tempfile import template

from langchain_core.prompts import PromptTemplate
"""
Template prompt terdiri dari template string. Template ini menerima serangkaian parameter dari pengguna yang dapat 
digunakan untuk menghasilkan prompt untuk model.

"non-message prompt" adalah jenis template prompt yang tidak memiliki struktur pesan tertentu, tetapi hanya terdiri dari string template yang dapat diisi dengan parameter.
'input_variables'
Dalam contoh ini, kita membuat template prompt yang menerima satu parameter, yaitu "topic". Template ini kemudian digunakan untuk menghasilkan prompt 
yang menjelaskan tentang topik tertentu secara singkat. Ketika kita memanggil metode invoke dengan memberikan nilai untuk parameter "topic", 
template akan menghasilkan prompt yang sesuai dengan format yang telah ditentukan.

Selain itu, kita juga dapat menggunakan 'partial_variables' untuk menetapkan nilai default untuk parameter tertentu dalam template. 
Dalam contoh kedua, kita menetapkan nilai default "Artificial Intelligence" untuk parameter "topic". Dengan menggunakan 'partial_variables', 
kita dapat membuat template yang lebih fleksibel dan mudah digunakan, karena kita tidak perlu selalu memberikan nilai untuk setiap parameter saat memanggil metode invoke.
"""

promptv1 = PromptTemplate(
    input_variables=["topic"],
    template="Jelaskan tentang {topic} secara singkat.",
)
print(promptv1.invoke({"topic": "Artificial Intelligence"}))
# print(prompt.invoke({"topic": ""}))

promptv2 = PromptTemplate(
    partial_variables={"topic": "Artificial Intelligence"},
    template="Jelaskan tentang {topic} secara singkat.",
)

print(promptv2.invoke({}))
"""
dalam contoh ketiga, kita menggunakan 'partial_variables' untuk menetapkan nilai default "secara sederhana..". Dengan cara ini, kita dapat 
menghasilkan prompt yang menjelaskan tentang topik tertentu dengan gaya penjelasan yang lebih sederhana. Ketika kita memanggil metode invoke 
dengan memberikan nilai untuk parameter "topic", template akan menghasilkan prompt yang sesuai dengan format yang telah ditentukan,
"""
promptv3 = PromptTemplate(
    input_variables=["topic"],
    partial_variables={"style": "secara sederhana seperti untuk anak-anak usia 5 tahun"},
    template="Jelaskan tentang {topic} {style}.",
)
print(promptv3.invoke({"topic": "Artificial Intelligence"}))



print("="*100)

"""
from message prompt
"message prompt" adalah jenis template prompt yang memiliki struktur pesan tertentu, biasanya terdiri dari beberapa bagian seperti "system", "user", dan "assistant". 
Struktur ini digunakan untuk mensimulasikan percakapan antara pengguna dan model, di mana setiap bagian memiliki peran tertentu dalam konteks percakapan.Dalam contoh pertama, kita membuat template prompt dengan struktur pesan yang terdiri dari bagian "system" dan "user". Bagian "system" memberikan
instruksi kepada model untuk menjelaskan tentang topik tertentu secara singkat, sementara bagian "user" 
memberikan nilai untuk parameter "topic". Ketika kita memanggil metode invoke dengan memberikan nilai untuk parameter "topic", template akan menghasilkan prompt yang sesuai dengan format yang telah ditentukan.
"""

from langchain_core.prompts import ChatPromptTemplate
promptv4 = ChatPromptTemplate.from_messages(
    [
        ("system", "Jelaskan tentang {topic} secara singkat."),
        ("human", "{topic}"),
        ("ai", "Hi, saya adalah asisten AI yang akan membantu menjelaskan tentang {topic}. {input}"),
        ("human","{input}"),
        
    ]
)

promptv5=promptv4.partial(topic="Artificial Intelligence")
messages = promptv5.format_messages(input="jelaskan dengan Bahasa yang Sederhana")
print("\nSemua pesan yang diformat:")
for msg in messages:
    print(f"{msg.type}: {msg.content}\n")

print(promptv4.invoke({"topic": "Artificial Intelligence", "input": "jelaskan dengan Bahasa yang Sederhana"}))


print("="*100)


"""
output parser adalah alat yang digunakan untuk memproses dan menginterpretasikan output yang dihasilkan oleh model.
Output parser dapat digunakan untuk mengambil informasi tertentu dari output model, seperti entitas, nilai, atau struktur data tertentu. 
Output parser dapat membantu dalam mengorganisir dan memahami hasil yang dihasilkan oleh model, 
sehingga memudahkan pengguna untuk mengambil informasi yang relevan dan berguna dari output tersebut.
Artinya parser membuat output AI bisa dipakai untuk keperluan lain, misalnya untuk membuat tabel, grafik, atau format 
data tertentu yang dapat digunakan dalam aplikasi atau analisis lebih lanjut.

"""

"""
Dalam contoh ini, kita menggunakan JsonOutputParser untuk memproses output yang dihasilkan oleh model. JsonOutputParser akan mencoba untuk menginterpretasikan output sebagai format JSON, sehingga kita dapat dengan mudah mengambil
informasi yang relevan dari output tersebut. Dalam contoh ini, kita menggunakan JsonOutputParser untuk mengambil jawaban yang dihasilkan oleh model dan menyimpannya dalam format JSON. Kita juga menggunakan ChatPromptTemplate 
untuk membuat template prompt yang akan digunakan untuk menghasilkan output dari model.
"""


from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import os
from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=getenv("MODEL"),
)
# parser = JsonOutputParser()
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Jelaskan tentang {topic} secara singkat. Jawab HANYA dengan 1 JSON object valid. {format_instructions}"),
#     ("ai", "Hi, saya adalah asisten AI yang akan membantu menjelaskan tentang {topic}. {input} {format_instructions}"),
#     ("human","{input}"),
# ]).partial(format_instructions=parser.get_format_instructions())

# chain = prompt | llm | parser

# result = chain.invoke({"topic": "Artificial Intelligence", "input": "jelaskan dengan Bahasa yang Sederhana"})
# filename = "output.json"
# try:
#     with open(filename, "r") as f:
#         data = json.load(f)
#     if not isinstance(data, list):
#         data = [data]
# except (FileNotFoundError, json.JSONDecodeError):
#     data = []
# data.append(result)
# with open(filename, "w") as f:
#     json.dump(data, f, indent=4)
# print(f"Hasil telah disimpan ke {filename}")
# print(result)




print("="*100)

"""
sructured output parser adalah jenis output parser yang dirancang untuk menghasilkan output yang tersesrutuk dan fleksibel (Structured output allows agents to 
return data in a specific, predictable format. This is useful for parsing the output of an agent and using it in a structured way, such as for filling out a form, 
creating a table, or generating code.) Structured output parser memungkinkan kita untuk menghasilkan output yang terstruktur dan fleksibel, 
sehingga kita dapat dengan mudah mengambil informasi yang relevan dari output
tersebut dan menggunakannya dalam format yang sesuai untuk keperluan lain.Dalam contoh ini, kita menggunakan 
PydanticOutputParser untuk memproses output yang dihasilkan oleh model. PydanticOutputParser memungkinkan 
kita untuk mendefinisikan model Pydantic yang sesuai dengan struktur data yang kita harapkan dari output model. 
Dengan menggunakan PydanticOutputParser, kita dapat dengan mudah mengambil informasi yang relevan dari output model dan menyimpannya dalam format yang sesuai untuk keperluan lain.
"""

from pydantic import BaseModel
from typing import Any



class Explanation(BaseModel):
    jawaban: Any
# parserv10 = PydanticOutputParser(pydantic_object=Explanation)
# promptv10 = ChatPromptTemplate.from_messages([
#     ("system", "Jelaskan tentang {topic} secara singkat"),
#     ("human", "{topic}"),
#     ("ai", "Hi, saya adalah asisten AI yang akan membantu menjelaskan tentang {topic}. {input} {format_instructions}"),
#     ("human","{input}"),
# ]).partial(format_instructions=parserv10.get_format_instructions())

# structured_chain = promptv10 | llm | parserv10
# result = structured_chain.invoke({"topic": "Artificial Intelligence", "input": "jelaskan dengan Bahasa yang Sederhana"})
# print(result.jawaban)
# print(type(result.jawaban))

print("="*100)

"""
Jsonoutputparser juga bisa di bind pydantic model ya, sama untuk yg menggungkan structure output coba kamu pelajari
with_structured_output, itu base function nya untuk chatmodel, itu bisa ngebuat 
ai ngeluarin object tanpa harus ada format instruction lagi

code dibawah ini adalah contoh penggunaan with_structured_output untuk menghasilkan output tanpa harus menggunakan format instruction lagi.
Dengan menggunakan with_structured_output, kita dapat langsung menghasilkan output dalam format yang sesuai dengan model Pydantic yang telah kita definisikan, 
sehingga memudahkan kita untuk mengambil informasi dan menggunakannya dalam format yang sesuai keperluan kita.
"""
promptv20 = ChatPromptTemplate.from_messages([
    ("system", "Jelaskan tentang {topic} secara singkat"),
    ("human", "{topic}"),
    ("ai", "Hi, saya adalah asisten AI yang akan membantu menjelaskan tentang {topic}. {input}"),
    ("human","{input}"),
])

# promptv30 = ChatPromptTemplate.from_messages([
#     ("system", "jawab pertanyaan matematika berikut {topic} secara singkat tanpa perlu penjelasan, langsung jawab saja"),
#     ("human", "{topic}"),
#     ("ai", "{topic}"),
#     # ("human","{input}"),
# ])

structured_llm = llm.with_structured_output(Explanation)
chain = promptv20 | structured_llm

result = chain.invoke({"topic": "Artificial Intelligence", "input": "jelaskan dengan Bahasa yang Sederhana"})

filename = "output.json"

try:
    with open(filename, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
except (FileNotFoundError, json.JSONDecodeError):
    data = []

data.append(result.model_dump())  

with open(filename, "w") as f:
    json.dump(data, f, indent=4)

print(f"Hasil telah disimpan ke {filename}")
print(result.jawaban)
print(type(result.jawaban))


# chain = promptv30 | structured_llm

# result2 = chain.invoke({"topic": "1+1"})

# print(result2.jawaban)
# print(type(result2.jawaban))
