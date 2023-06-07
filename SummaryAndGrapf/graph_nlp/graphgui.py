import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from rouge import Rouge
from operator import itemgetter
import csv
import nltk
import torch
import gensim
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

def dosyacevir(dosyaYolu):
    nltk.download('punkt')

    with open(dosyaYolu, 'r') as file:
        lines = file.readlines()

    
    cumleler = []
    for satir in lines:
        satir = satir.strip()  
        if satir: 
            cumleler.extend(nltk.tokenize.sent_tokenize(satir))

    with open('cumleler.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["cümleler"])  
        for cumle in cumleler:
            writer.writerow([cumle])  


def Onİsleme():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('porter_test')

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    def preprocess(cumle):
        kelimeler = word_tokenize(cumle)
        kelimeler = [kelime for kelime in kelimeler if kelime.isalnum()]
        kelimeler = [kelime for kelime in kelimeler if not kelime in stop_words]
        kelimeler = [ps.stem(kelime) for kelime in kelimeler]
        return ' '.join(kelimeler)

    with open('cumleler.csv', 'r') as file:
        okuma = csv.reader(file)
        cumleler = list(okuma)[1:]

    processed_sentences = [preprocess(cumle[0]) for cumle in cumleler]

    with open('islenmis_cumleler.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["cümleler"])
        for cumle in processed_sentences:
            writer.writerow([cumle])


def WordEmbeddingYontemi():
    nltk.download('punkt')

    with open('islenmis_cumleler.csv', 'r') as file:
        okuma = csv.reader(file)
        cumleler = list(okuma)[1:]

    islenmisVeri = [word_tokenize(cumle[0]) for cumle in cumleler]

    model = gensim.models.Word2Vec(sentences=islenmisVeri, vector_size=100, window=5, min_count=1, workers=4)

    islenmis_vektorler = []
    for cumle in islenmisVeri:
        vector = sum(model.wv[kelime] for kelime in cumle if kelime in model.wv) / len(cumle)
        islenmis_vektorler.append(vector)
    
    return islenmis_vektorler


def BertYontemi():
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open('islenmis_cumleler.csv', 'r') as file:
        okuma = csv.reader(file)
        cumleler = list(okuma)[1:]

    islenmis_vektorler = []
    for cumle in cumleler:
        inputs = tokenizer(cumle[0], return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states[-1]
        sentence_vector = torch.mean(hidden_states, dim=1).squeeze().detach().numpy()
        islenmis_vektorler.append(sentence_vector)
    return islenmis_vektorler

def OzelİsimYazdirma(cumle):
    ozelİsim = 0
    kelimeler = cumle.split()
    for kelime in kelimeler:
        if kelime.istitle():
            ozelİsim += 1
    ozelİsim -= 1
    return ozelİsim

def NumerikVeriSaydirma(cumle):
    numerik_veri = 0
    kelimeler = cumle.split()
    for kelime in kelimeler:
        if kelime.isnumeric():
            numerik_veri += 1
    return numerik_veri
 
def BaslikSaydirma(cumle,ilk_satir):
    cumle_kelimeleri = cumle.split()
    baslik_kelimeleri = ilk_satir.split()

    baslikta_gecen_kelime_sayisi = 0

    for kelime in baslik_kelimeleri:
        if kelime in cumle_kelimeleri:
            baslikta_gecen_kelime_sayisi += 1
    return baslikta_gecen_kelime_sayisi

def TemaKelimeSaydirma(cumle, dokuman):
    vektorleyici = TfidfVectorizer()
    tfidf_matrisi = vektorleyici.fit_transform([dokuman, cumle])
    ozellik_isimleri = vektorleyici.get_feature_names_out()

    toplam_kelime_sayisi = len(ozellik_isimleri)
    tema_kelime_limiti = int(toplam_kelime_sayisi * 0.1)

    tema_kelimeler = []

    for i, kelime in enumerate(ozellik_isimleri):
        tfidf_degeri = tfidf_matrisi[1, i]
        
        if tfidf_degeri > 0:
            tema_kelimeler.append(kelime)
        
        if len(tema_kelimeler) >= tema_kelime_limiti:
            break

    return len(tema_kelimeler)

class Application(tk.Tk):

    def DosyaGetir(self):
        self.dosyaYolu = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        dosyacevir(self.dosyaYolu)
        Onİsleme()

    def __init__(self):
        super().__init__()
        self.title('Cümle Benzerlik Hesaplayıcı')
        self.geometry('1000x1000')

        self.dosyaYolu = None

        self.method_var = tk.StringVar()
        self.graph_var = tk.StringVar()
        self.benzerlik_thresholdu = tk.StringVar()
        self.skor_thresholdu = tk.StringVar()

        self.canvas = None
        self.toolbar = None

        self.create_widgets()

  

    def create_widgets(self):

        panel_frame = tk.Frame(self, borderwidth=2, relief=tk.GROOVE, padx=10, pady=10)
        panel_frame.pack(side=tk.LEFT, fill=tk.Y)
   
        tk.Label(panel_frame, text="Hangi yöntemi istersiniz?").pack()
       
        tk.Checkbutton(panel_frame, text="Word Embedding", variable=self.method_var, onvalue='1', offvalue='').pack(anchor=tk.W)
        tk.Checkbutton(panel_frame, text="BERT", variable=self.method_var, onvalue='2', offvalue='').pack(anchor=tk.W)
       
        self.dosyaButonu = tk.Button(panel_frame, text="Başlangıç dosyasını seç", command=self.DosyaGetir)
        self.dosyaButonu.pack(pady=10)

        tk.Label(panel_frame, text="Cümle Benzerliği Threshold Değeri:").pack(anchor=tk.W)
        self.threshold_entry = tk.Entry(panel_frame, textvariable=self.benzerlik_thresholdu)
        self.threshold_entry.pack()

        tk.Label(panel_frame, text="Cümle Skoru Threshold Değeri:").pack(anchor=tk.W)
        self.threshold_entry = tk.Entry(panel_frame, textvariable=self.skor_thresholdu)
        self.threshold_entry.pack()

        self.olustur = tk.Button(panel_frame, text="Oluştur", command=self.calculate)
        self.olustur.pack(pady=10)
        
        self.ozetOlustur = tk.Button(panel_frame, text="Özet Oluştur", state=tk.DISABLED, command=self.open_summary_page)
        self.ozetOlustur.pack()

        self.ozetOlustur.pack()

        self.canvas = None
        self.toolbar = None

        
  

    def calculate(self):
        method = self.method_var.get()

        if self.dosyaYolu is None:
            tk.messagebox.showerror("Error", "Başlangıç dosyası seçmediniz!")
            return

        if method == '1':
            islenmis_vektorler = WordEmbeddingYontemi()
        elif method == '2':
            islenmis_vektorler = BertYontemi()
        else:
            tk.messagebox.showerror("Error", "Invalid input, exiting...")
            return

        if len(islenmis_vektorler) > 1:
            threshold = float(self.benzerlik_thresholdu.get())
            with open('islenmis_cumleler.csv', 'r') as file:
                okuma = csv.reader(file)
                cumleler = list(okuma)[1:]
            self.GrafCizdirme(cumleler, islenmis_vektorler, threshold)
          
        else:
            tk.messagebox.showinfo("Result", "Not enough cumleler for similarity calculation.")
        
    def GrafCizdirme(self, cumleler, islenmis_vektorler, threshold):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar is not None:
            self.toolbar.destroy()
        G = nx.Graph()

        node_degrees = {}
        node_skor={}
        node_toplam_baglanti={}

        for i in range(len(cumleler)):
            node_name = "Cumle"+str(i+1)
            G.add_node(node_name)
            node_degrees[node_name] = 0  
            node_toplam_baglanti[node_name]=0

        for i in range(len(cumleler)):
            for j in range(i + 1, len(cumleler)):
                similarity = cosine_similarity(islenmis_vektorler[i].reshape(1, -1), islenmis_vektorler[j].reshape(1, -1))[0][0]
                node1 = "Cumle"+str(i+1)
                node2 = "Cumle"+str(j+1)
                node_toplam_baglanti[node1] +=1
                node_toplam_baglanti[node2] +=1

                if similarity > threshold:
                    node1 = "Cumle"+str(i+1)
                    node2 = "Cumle"+str(j+1)
                    G.add_edge(node1, node2, weight=round(similarity, 2))
                    # Node bağlantı sayısını artır
                    node_degrees[node1] += 1
                    node_degrees[node2] += 1
                


        fig = plt.figure(figsize=(10,10))
        pos = nx.spring_layout(G, seed=42)

        nx.draw(G, pos, with_labels=True, font_size=8, node_color='turquoise', node_size=1000)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        for node, (x, y) in pos.items():
                plt.text(x, y+0.1, s=f"{node_degrees[node]}", bbox=dict(facecolor='yellow', alpha=0.5))

        self.scores = []
        
        with open('cumleler.csv', 'r') as file:
            okuma = csv.reader(file)
            islenmemis_cumleler = [row[0] for row in okuma][1:]
        
        with open('islenmis_cumleler.csv', 'r') as file:
            okuma = csv.reader(file)
            islenmis_cumleler = [row[0] for row in okuma][1:]
        self.document = '. '.join(islenmis_cumleler)
        ilk_satir = islenmis_cumleler[0]
        i=0
        for islenmis_cumle in islenmis_cumleler:
            node_name = "Cumle"+str(i+1)
            score = self.calculate_sentence_score(islenmis_cumle,islenmemis_cumleler[i] ,self.document,node_degrees[node_name],node_toplam_baglanti[node_name],ilk_satir)
            self.scores.append((islenmemis_cumleler[i], score))
            node_skor[node_name]=score
            i=i+1
        self.ozetOlustur.configure(state=tk.NORMAL)
        threshold1 = round(float(self.skor_thresholdu.get()),2)
        
        for node, (x, y) in pos.items():
            if node_skor[node] > threshold1 :
                plt.text(x + 0.05,y+0.1 , s=f"{node_skor[node]}", bbox=dict(facecolor='red', alpha=0.5))
            else :
                plt.text(x + 0.05,y+0.1, s=f"{node_skor[node]}", bbox=dict(facecolor='green', alpha=0.5))
        
        self.canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1) 

    
    def calculate_sentence_score(self,islenmis_cumle,islenmemis_cumle, document,treshold,toplam,ilk_satir):

        ozelİsim = OzelİsimYazdirma(islenmemis_cumle) / len(islenmemis_cumle)

        numerik_veri = NumerikVeriSaydirma(islenmis_cumle) / len(islenmis_cumle)

        baglantiSayisi = treshold / toplam
        
        baslik_kelimeleri = BaslikSaydirma(islenmis_cumle, ilk_satir) / len(islenmis_cumle)

        temakelimeler = TemaKelimeSaydirma(islenmis_cumle, document) / len(islenmis_cumle)

        # Cümle skoru hesaplama
        sentence_score = (ozelİsim +  numerik_veri +  baglantiSayisi +  baslik_kelimeleri + temakelimeler)/5
        sentence_score=round(sentence_score,2)
        return sentence_score  
      
    def open_summary_page(self):
        summary_window = tk.Toplevel(self)

        ozetMetin = tk.Label(summary_window, text="Özet:")
        ozetMetin.pack()

        ozetAlani = tk.Text(summary_window, height=15, width=170)
        ozetAlani.insert(tk.END, self.Ozetleme())
        ozetAlani.pack()


        metin = tk.Label(summary_window, text="Metin:")
        metin.pack()

        metinAlani = tk.Text(summary_window, height=15, width=170)
        metinAlani.pack()

        

        olustur = tk.Button(summary_window, text="Skor Hesapla", command=lambda: self.RougeSkor(ozetAlani.get("1.0", tk.END), metinAlani.get("1.0", tk.END)))
        olustur.pack()

        self.ozetMetin = tk.Label(summary_window, text="")
        self.ozetMetin.pack()

    def RougeSkor(self, ozet, reference):
        rouge = Rouge()
        scores = rouge.get_scores(ozet, reference)
        rouge_score = scores[0]['rouge-l']['f']
        self.OzetSkorGuncelle(rouge_score)

    def OzetSkorGuncelle(self, rouge_score):
        self.ozetMetin.config(text="Rouge Skoru: " + str(rouge_score))

    def Ozetleme(self):
        self.scores = sorted(self.scores, key=itemgetter(1), reverse=True)
        sayi=int(len(self.scores)/2)

        secilenKelimeler = [cumle for cumle, score in self.scores[1:sayi]]

        ozet = ' '.join(secilenKelimeler)
        return ozet
        
     
            

    

if __name__ == "__main__":
    app = Application()
    app.mainloop()
