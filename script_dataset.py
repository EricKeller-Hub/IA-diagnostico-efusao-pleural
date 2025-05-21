import os
import shutil
import pandas as pd 

# Caminho para o CSV
csv_path = "C:/Users/keller/Desktop/effusion_vs_normal_balanced.csv"

# Caminho da pasta que contém TODAS as imagens
origem_imagens = "C:/Users/keller/Desktop/images"

# Caminhos de destino
destino_base = "C:/Users/keller/Desktop/dataset"
destino_efusao = os.path.join(destino_base, "effusion")
destino_normal = os.path.join(destino_base, "normal")

# Criar diretórios se não existirem
os.makedirs(destino_efusao, exist_ok=True)
os.makedirs(destino_normal, exist_ok=True)

# Carregar o CSV
df = pd.read_csv(csv_path)

# Copiar as imagens
copiados = 0
for _, row in df.iterrows():
    nome_arquivo = row["Image Index"]
    label = row["Effusion_Label"]
    
    origem = os.path.join(origem_imagens, nome_arquivo)
    
    if not os.path.exists(origem):
        print(f"Imagem não encontrada: {origem}")
        continue
    
    destino = destino_efusao if label == 1 else destino_normal
    shutil.copy2(origem, os.path.join(destino, nome_arquivo))
    copiados += 1

print(f"{copiados} imagens copiadas com sucesso!")
