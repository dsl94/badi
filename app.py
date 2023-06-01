from fastapi import Request, FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import openai
import os
import pandas as pd
from scipy import spatial
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime

class Question(BaseModel):
    question: str

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = openai.api_key = os.getenv("OPENAI_API_KEY")
cred = credentials.Certificate('firebase-sdk.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def writeToFirestore(question, answer):
    doc_ref = db.collection('badi').document(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    doc_ref.set({

        'question': question,
        answer: answer,
    })

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to this fantastic app!"}

@app.post("/ask")
async def search(question: Question):
    df = pd.DataFrame()
    benefiti = """Vodeći se idejom da su ljudi na prvom mestu, Badin Soft omogućuje svojim zaposlenima brojne benefite od kojih izdvajamo

        Privatno zdravstveno osiguranje

        Pravo na uključenje u Dunav privatno zdravstveno osiguranje naši zaposleni stiču nakod 3 meseca rada u Badinu.

        Osnovni paket u koji su uključeni naši zaposleni i poslovni saradnici obuhvata pokriće troškova do 5000€ godišnje i tu se ubrajaju: ambulantno lečenje, dijagnostičke procedure, lečenje u zdravstvenoj ustanovi sekundarnog ili tercijarnog nivoa-bolnice i klinički centri Srbije (do 30 dana godišnje), hirurške intervencije, prepisane lekove, fizikalne, psihijatrijskog lečenja, homeopatije i akupunkture, stomatolog i oftamolog (stakla i okviri), logoped kao i lečenja hroničnih oboljenja, zdravstvenu zaštitu trudnica i troškove porođaja.

        U sklopu polise zdravstvenog osiguranje imate ugovoreno pokriće sistematskog pregleda koji se održava jednom godišnje za ceo kolektiv.

        Dogovorena karenca je 0, tj. osiguranje počinje da važi odmah po uključivanju u osiguranje.

        Detaljnije o privatnom zdravstvenom osiguranju možete pročitati ovde

        BizMix

        Pravo na tarifni paket, čije troškove pokriva Badin Soft , ili umreženje privatnog broja (ukoliko nije pod ugovornom obavezom kod drugog operatera) stiču svi zaposleni u Badinu nakon 3 meseca od stupanja u radni odnos. Takođe, nakon ovog perioda, moguće je uzeti mobilni aparat. Mobilni aparat se otplaćuje u ratama koje se odbijaju od mesečne zarade. Ovo pravo je opciono i molimo sve koji žele da ga iskoriste da se za sve dalje informacije obrate OFA ili HR sektoru.

        Detalje o tarifama i naručivanju uređaja možete pročitati ovde.

        Administrativne zabrane

        Badin Soft ima potpisane ugovore sa firmama iz uslužnih delatnosti kako bi svojim zaposlenima prižio mogućnost kupovine i otplate određenih dobara na rate.

        Kako bi mogli da iskoristite ove povoljnosti potrebno je kontaktirati nekog iz OFA ili finansijskog sektora kako bi vam izdali potvrde o zaposlenju i detaljnije vas uputili u proceduru.

        Firme sa kojima imamo potpisane ugovore su:

     -Turistična agencija „Generacija 77“ što znači da ćete biti u mogućnosti da odmor isplaćujete u mesečnim ratama bez kamate (od dve do   devet rata). Generacija 77 radi po principu subagenta, tako da na raspolaganju imate veliki izbor aranžmana raznih drugih agencija, čiju listu   možete pogledati ovde Agencija se nalazi u Kalči, Lokal 68a/c​​​​​​​

     -Capriolo
    Radnja za prodaju bicikla, delova za bicikle i fitnes opreme. Robu iz njihovog asortimana možete isplaćivati u šest, devet ili dvanaest rata bez kamate.

    -Tehnomanija
    Tehničku robu u isnosima od 10.000 d0 150.000, možete isplaćivati do 12 mesečnih rata u prodajnim objektima Tehnomanije.

    -Forma Ideale
     Robu iz asortimana “Forma Ideale” možete kupiti sa odloženim plaćanjem na tri, šest ili dvanaest rata, s tim da iznos rate ne može biti veći od    30% mesečne zarade zaposlenog.

    -Emmezeta
    Robu iz asortimana “Emmezeta” možete kupiti sa odloženim plaćanjem na šest  rata, s tim da iznos rate ne može biti veći od  30% mesečne zarade zaposlenog.

    -Intersport
    Robu iz asortimana sportske opreme "Intersport" možete kupiti sa odloženim paćanjem do 9 mesećenih rata, s tim da iznos ne može biti veći od 30% mesečne zarade zaposlenog.

    -Office shoes
    Robu iz asortimana "Office shoes"-a možete kupiti na 5 mesečnih rata za iznose preko 10.000 dinara.

        Refundacija za sport

    U cilju promovisanja zdravog načina života i timskih vrednosti, Badin je obezbedio pogodnosti za sve one koje žele da se bave nekim sportskim aktivnostima. Badin Soft refundira mesečne članarine u svim sportskim klubovima ili teretanama u iznosu do 1500 dinara (dokaz o uplati članarine potrebno je odneti svakog meseca u OFA sektor). Umesto pojedinačnih aktivnosti, za one kolege koje žele da se bave grupnim sportovima Badin finansira zakup košarkaških i fudbalskih sala. Svi koji su zainteresovani za ovaj vid rekreacije, potrebno je da obaveste OFA sektor i dogovore detalje.

       """

    query = f"""Use the below article to answer questions about Badin in serbian, if you don't know the answer write "Na žalost ne znam odgovor, molim te da se obratis HR ili OFA službi."
    Article:
    \"\"\"
    {benefiti}
    \"\"\"
    Question: {question.question}?"""

    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': 'You answer questions about benefits'},
            {'role': 'user', 'content': query},
        ],
        model=GPT_MODEL,
        temperature=0,
    )

    result = {
        "answer": response['choices'][0]['message']['content']
    }
    writeToFirestore(question.question, result['answer'])
    return result


@app.post("/recipe")
async def search(question: Question):
    prompt = f"""Can you give me a recipe with this ingredients: {question.question}"""
    # prompt = f"""Can you give me a recipe with this ingredients: beef, cheese, garlic, gnocchi?"""
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=1000)
    food = response['choices'][0]['text']
    print(food)
    short = f"""Make this recipe shorter so it can be used as prompt for image generation using dalle {food}?"""
    response2 = openai.Completion.create(model="text-davinci-003", prompt=short, temperature=0, max_tokens=1000)
    food2 = response2['choices'][0]['text']
    food_prompt = "Photorealistic Image of food in a plate with this recipe: " + food2
    response = openai.Image.create(
        prompt=food_prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']

    return {
        "recipe": food,
        "image_url": image_url
    }
