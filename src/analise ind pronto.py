from ultralytics import YOLO
import cv2
import time
import serial

# Carrega o modelo
model = YOLO(r"C:\Users\yan.fernandes\Downloads\weights.pt")

# Inicia a câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao tentar acessar a câmera. Verifique se está conectada corretamente.")
    exit()

# Parâmetros de validação
CLASSE_DE_ESTRIBO = "Estribo"
area_x1, area_y1 = 220, 540
area_x2, area_y2 = 500, 580

# Tolerâncias
tolerancia_posicao = 75  # Para afrouxar as regras de posição
tolerancia_rotacao = 0.75  # Para permitir mais variação no ângulo

# Aspect ratio esperado baseado na bounding box ideal
altura_esperada = area_y2 - area_y1
largura_esperada = area_x2 - area_x1
proporcao_ideal = largura_esperada / altura_esperada

# Configuração da porta do relé (ajuste conforme necessário)
try:
    porta_rele = 'COM5'  # No caso de conexão serial no Windows
    conexao_serial = serial.Serial(porta_rele, 9600, timeout=1)
except Exception as e:
    print(f"Erro ao conectar na porta {porta_rele}: {e}")
    exit()

# Função para verificar se valor está dentro da tolerância
def dentro_tolerancia(valor, esperado, margem):
    return (esperado - margem) <= valor <= (esperado + margem)

# Função para verificar se há anomalias no Estribo
def verificar_estribo(x1, y1, x2, y2):
    largura = x2 - x1
    altura = y2 - y1 + 0.01  # evitar divisão por zero
    aspect_ratio = largura / altura

    erros = []

    if not dentro_tolerancia(x1, area_x1, tolerancia_posicao):
        erros.append("Posição X fora do esperado")
    if not dentro_tolerancia(y1, area_y1, tolerancia_posicao):
        erros.append("Posição Y fora do esperado")
    if abs(aspect_ratio - proporcao_ideal) > tolerancia_rotacao:
        erros.append("Angulação suspeita (Estribo torto)")

    return erros

# Funções para controlar o relé (sirene)
def ativar_rele():
    conexao_serial.write(bytes([0xA0, 0x01, 0x01, 0xA2]))  # Ativa o relé
    print("Relé ativado (sirene ligada)!")
    global hora_ativacao
    hora_ativacao = time.time()  # Armazena o momento da ativação

def desativar_rele():
    conexao_serial.write(bytes([0xA0, 0x01, 0x00, 0xA1]))  # Desativa o relé
    print("Relé desativado (sirene desligada)!")

# Função para verificar se o relé está ativo e desativá-lo
def verificar_e_desativar_rele():
    # Aqui você pode adicionar o comando necessário para verificar o estado atual do relé.
    # Supondo que o relé seja um dispositivo simples, e você saiba o comando que indica
    # o estado, caso contrário, você pode simplesmente desativar logo no início.
    # Para efeito de exemplo, vamos desativá-lo sempre.

    desativar_rele()  # Desativa o relé se ele estiver ativo (ou não).

# Inicializa as variáveis de contagem de detecções
tela_detectada = 0
estribo_detectado = 0
ultimo_estado_rele = None  # Para verificar mudanças no estado do relé

# Variáveis para controle do tempo
tempo_ativacao_rele = 3  # Tempo em segundos que o relé ficará ativo
hora_ativacao = None  # Armazenará o momento da ativação do relé

print("Sistema de verificação de Estribo INICIADO.")
print(model.names)

# Verifica o estado do relé ao iniciar
verificar_e_desativar_rele()

# Limite de confiança para as classes
limite_confianca_estribo = 0.85  # Limite de confiança para "Estribo"
limite_confianca_tela = 0.90  # Limite de confiança para "Tela"
# Contadores para Tela e Estribo
tela_detectada = 0
estribo_detectado = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro na leitura da câmera.")
        break

    results = model(frame)[0]

    # Desenha a área de anomalia (sempre visível)
    cv2.rectangle(frame, (area_x1, area_y1), (area_x2, area_y2), (255, 0, 255), 2)
    cv2.putText(frame, "Area de Anomalia", (area_x1, area_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Verifica as detecções
    for box in results.boxes:
        nome_classe = model.names[int(box.cls)]
        conf = float(box.conf)

        # Detecta "Tela" com confiança > 0.90
        if nome_classe == "Tela" and conf > limite_confianca_tela:
            tela_detectada += 1
            # Desenha a caixa da Tela
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"{nome_classe} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Detecta "Estribo" com confiança > 0.85
        elif nome_classe == "Estribo" and conf > limite_confianca_estribo:
            estribo_detectado += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cor_estribo = (0, 255, 0)  # Verde por padrão
            erros = verificar_estribo(x1, y1, x2, y2)

            if erros:
                cor_estribo = (0, 0, 255)  # Vermelho para anomalia
                if ultimo_estado_rele != "ativo":  # Só ativa o relé se ele ainda não estiver ativo
                    ativar_rele()
                    ultimo_estado_rele = "ativo"
            else:
                if ultimo_estado_rele != "inativo":  # Só desativa o relé se ele ainda não estiver inativo
                    desativar_rele()
                    ultimo_estado_rele = "inativo"

            cv2.rectangle(frame, (x1, y1), (x2, y2), cor_estribo, 2)
            label = f"{nome_classe} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_estribo, 2)

            # Exibe erros no terminal se houver anomalia
            if erros:
                print(f"⚠️ ANOMALIA DETECTADA: {', '.join(erros)}")

    # Verifica se há mais de 1 tela ou estribo
    if tela_detectada > 1:
        print("⚠️ Mais de 1 Tela detectada!")
        ativar_rele()

    if estribo_detectado > 1:
        print("⚠️ Mais de 1 Estribo detectado!")
        ativar_rele()

    # Verifica o tempo de ativação do relé e desativa após o tempo determinado
    if hora_ativacao and (time.time() - hora_ativacao) > tempo_ativacao_rele:
        desativar_rele()  # Desativa o relé após o tempo especificado
        hora_ativacao = None  # Reseta o tempo de ativação

    # Exibe a imagem com as caixas desenhadas
    cv2.imshow("Monitoramento de Estribos", frame)

    if cv2.waitKey(1) & 0xFF == ord("p"):
        break

cap.release()
cv2.destroyAllWindows()
