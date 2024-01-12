import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import requests
from flask import Flask

app = Flask(__name__)

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://carcinamon-comunity-default-rtdb.asia-southeast1.firebasedatabase.app/'
})


@app.route("/", methods=["GET","POST"])
def index():
    try:
        # Mendapatkan referensi ke Firebase Realtime Database
        ref = db.reference('post')

        # Membaca data terbaru di bawah "post"
        latest_post = ref.order_by_key().limit_to_last(1).get()

        # Mengekstrak nilai entri terbaru
        for key, value in latest_post.items():
            print("ID: ", key)
            entry_id = key
            judul = value['header']
            text = value['text']

        resp = requests.post("https://model-ml-7vvpza7dfa-et.a.run.app", data={'sentence': text}) #ganti url sama yang API ML Model
        resp2 = requests.post("https://model-ml-7vvpza7dfa-et.a.run.app", data={'sentence': judul}) #ganti url sama yang API ML Model

        json_response = resp.json()
        json_respone2 = resp2.json()
        prediction = json_response.get('prediction')
        prediction2 = json_respone2.get('prediction')

        print(json_response)
        #print('Hasil prediksi:', prediction)

        if prediction > 0.5 or prediction2 > 0.5: #treshold untuk dianggap sebagai kalimat/paragraf toxic
            ref.child(entry_id).delete()
            print("dokumen di hapus karena toxic")

    except requests.exceptions.JSONDecodeError as e:
        print("Terjadi kesalahan dalam memparsing respons JSON:", str(e))

    return "OK"

if __name__ == "__main__":
    app.run(debug=True)