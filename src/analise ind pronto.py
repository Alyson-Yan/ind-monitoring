"""
Sistema de Verificação de Estribos com YOLO
Monitora a posição e angulação de estribos em tempo real
"""

from ultralytics import YOLO
import cv2
import time
import serial
import logging
from dataclasses import dataclass
from typing import List, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def diag(msg, *args):
    """Pequena função de diagnóstico: registra em logger.debug e imprime no stdout."""
    try:
        text = msg.format(*args) if args else str(msg)
    except Exception:
        try:
            text = f"{msg} {' '.join(map(str, args))}"
        except Exception:
            text = str(msg)
    try:
        logger.debug(text)
    except Exception:
        pass
    try:
        print(f"[DIAG] {text}")
    except Exception:
        pass


@dataclass
class AreaAnomaliaConfig:
    """Configuração da área de detecção"""
    x1: int = 220
    y1: int = 540
    x2: int = 500
    y2: int = 580
    
    @property
    def altura(self) -> int:
        return self.y2 - self.y1
    
    @property
    def largura(self) -> int:
        return self.x2 - self.x1
    
    @property
    def proporcao_ideal(self) -> float:
        return self.largura / self.altura


@dataclass
class ToleranciaConfig:
    """Configuração de tolerâncias"""
    posicao: int = 75  # pixels
    rotacao: float = 0.75  # aspect ratio
    confianca_estribo: float = 0.85
    confianca_tela: float = 0.90


class ReleControlador:
    """Gerencia comunicação serial com o relé"""
    
    COMANDO_LIGAR = bytes([0xA0, 0x01, 0x01, 0xA2])
    COMANDO_DESLIGAR = bytes([0xA0, 0x01, 0x00, 0xA1])
    
    def __init__(self, porta: str = 'COM5', baudrate: int = 9600):
        self.porta = porta
        self.baudrate = baudrate
        self.conexao = None
        self.ativo = False
        self.hora_ativacao = None
        self._conectar()
        diag("ReleControlador inicializado: porta=%s baudrate=%s", self.porta, self.baudrate)
    
    def _conectar(self) -> None:
        """Estabelece conexão serial com o relé"""
        try:
            self.conexao = serial.Serial(self.porta, self.baudrate, timeout=1)
            logger.info(f"Conexão serial estabelecida na porta {self.porta}")
            diag("Conexão serial estabelecida na porta %s", self.porta)
        except Exception as e:
            logger.error(f"Erro ao conectar na porta {self.porta}: {e}")
            diag("Erro ao conectar na porta %s: %s", self.porta, e)
            raise
    
    def ligar(self) -> None:
        """Ativa o relé (sirene)"""
        try:
            if not self.ativo:
                self.conexao.write(self.COMANDO_LIGAR)
                self.ativo = True
                self.hora_ativacao = time.time()
                logger.info("🔴 Relé ativado (sirene ligada)")
                diag("Relé ativado: porta=%s hora=%s", self.porta, self.hora_ativacao)
        except Exception as e:
            logger.error(f"Erro ao ativar relé: {e}")
            diag("Erro ao ativar relé: %s", e)
    
    def desligar(self) -> None:
        """Desativa o relé (sirene)"""
        try:
            if self.ativo:
                self.conexao.write(self.COMANDO_DESLIGAR)
                self.ativo = False
                self.hora_ativacao = None
                logger.info("🟢 Relé desativado (sirene desligada)")
                diag("Relé desativado: porta=%s", self.porta)
        except Exception as e:
            logger.error(f"Erro ao desativar relé: {e}")
            diag("Erro ao desativar relé: %s", e)
    
    def desligar_apos_timeout(self, timeout_segundos: int) -> None:
        """Desativa o relé após timeout"""
        if self.hora_ativacao and (time.time() - self.hora_ativacao) > timeout_segundos:
            self.desligar()
    
    def fechar(self) -> None:
        """Fecha a conexão serial"""
        if self.conexao and self.conexao.is_open:
            self.desligar()
            self.conexao.close()
            diag("Conexão serial fechada: porta=%s", self.porta)


class AnalisadorEstribo:
    """Verifica anomalias na detecção de estribos"""
    
    def __init__(self, area_config: AreaAnomaliaConfig, tol_config: ToleranciaConfig):
        self.area = area_config
        self.tolerancia = tol_config
    
    def _dentro_tolerancia(self, valor: float, esperado: float, margem: float) -> bool:
        """Verifica se valor está dentro da tolerância"""
        return (esperado - margem) <= valor <= (esperado + margem)
    
    def verificar_anomalias(self, x1: int, y1: int, x2: int, y2: int) -> List[str]:
        """
        Verifica se há anomalias na posição/angulação do estribo
        
        Args:
            x1, y1: Coordenadas superiores-esquerda
            x2, y2: Coordenadas inferiores-direita
            
        Returns:
            Lista de erros encontrados (vazia se válido)
        """
        erros = []
        
        # Calcular dimensões
        largura = x2 - x1
        altura = max(y2 - y1, 1)  # Evitar divisão por zero
        aspect_ratio = largura / altura
        
        # Validar posição X
        if not self._dentro_tolerancia(x1, self.area.x1, self.tolerancia.posicao):
            erros.append(f"Posição X fora: {x1} (esperado: {self.area.x1}±{self.tolerancia.posicao})")
        
        # Validar posição Y
        if not self._dentro_tolerancia(y1, self.area.y1, self.tolerancia.posicao):
            erros.append(f"Posição Y fora: {y1} (esperado: {self.area.y1}±{self.tolerancia.posicao})")
        
        # Validar angulação (aspect ratio)
        desvio_ratio = abs(aspect_ratio - self.area.proporcao_ideal)
        if desvio_ratio > self.tolerancia.rotacao:
            erros.append(f"Angulação suspeita (ratio: {aspect_ratio:.2f}, esperado: {self.area.proporcao_ideal:.2f})")
        
        return erros


class MonitorEstribo:
    """Orquestra o monitoramento de estribos"""
    
    def __init__(self, modelo_path: str, porta_rele: str = 'COM5'):
        self.rele = ReleControlador(porta_rele)
        self.modelo = YOLO(modelo_path)
        self.analisador = AnalisadorEstribo(AreaAnomaliaConfig(), ToleranciaConfig())
        
        # Estado
        self.contador_tela = 0
        self.contador_estribo = 0
        self.tempo_rele_ativo = 3  # segundos
        
        # Câmera
        self.camera = self._inicializar_camera()
        diag("MonitorEstribo inicializado: modelo=%s porta_rele=%s", modelo_path, porta_rele)
    
    def _inicializar_camera(self) -> cv2.VideoCapture:
        """Inicializa a câmera"""
        diag("Inicializando câmera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Erro ao acessar a câmera. Verifique a conexão.")
            diag("Erro ao acessar a câmera: dispositivo indisponível")
            raise RuntimeError("Câmera não disponível")
        logger.info("Câmera inicializada com sucesso")
        diag("Câmera inicializada com sucesso")
        return cap
    
    def processar_frame(self, frame) -> Tuple[int, int, bool]:
        """
        Processa um frame da câmera
        
        Returns:
            (contador_tela, contador_estribo, anomalia_detectada)
        """
        tela_frame = 0
        estribo_frame = 0
        anomalia_detectada = False
        diag("Processando frame: inicial")
        
        # Executar detecção
        resultados = self.modelo(frame)[0]
        try:
            num_boxes = len(resultados.boxes)
        except Exception:
            num_boxes = 0

        diag("Detecções no frame: {}", num_boxes)

        # Desenhar área de anomalia
        self._desenhar_area_anomalia(frame)

        # Processar detecções
        for i, deteccao in enumerate(resultados.boxes):

            nome_classe = self.modelo.names[int(deteccao.cls)]
            confianca = float(deteccao.conf)

            x1, y1, x2, y2 = map(int, deteccao.xyxy[0])

            diag(
                "Detecção {} -> Classe={} Conf={:.3f} BBox=({}, {}, {}, {})",
                i,
                nome_classe,
                confianca,
                x1,
                y1,
                x2,
                y2
            )

            if nome_classe == "Tela" and confianca > self.analisador.tolerancia.confianca_tela:
                tela_frame += 1
                self._desenhar_tela(frame, deteccao)

            elif nome_classe == "Estribo" and confianca > self.analisador.tolerancia.confianca_estribo:
                estribo_frame += 1

                anomalia = self._processar_estribo(frame, deteccao)

                if anomalia:
                    anomalia_detectada = True

        # Verificar múltiplas detecções
        if tela_frame > 1:
            logger.warning("⚠️ Múltiplas telas detectadas")
            anomalia_detectada = True

        if estribo_frame > 1:
            logger.warning("⚠️ Múltiplos estribos detectados")
            anomalia_detectada = True

        diag(
            "Processamento do frame finalizado: tela={} estribo={} anomalia={}",
            tela_frame,
            estribo_frame,
            anomalia_detectada
        )

        return tela_frame, estribo_frame, anomalia_detectada
    
    def _desenhar_area_anomalia(self, frame) -> None:
        """Desenha a área de detecção de anomalias"""
        area = self.analisador.area
        cv2.rectangle(frame, (area.x1, area.y1), (area.x2, area.y2), (255, 0, 255), 2)
        cv2.putText(frame, "Area de Anomalia", (area.x1, area.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    def _desenhar_tela(self, frame, deteccao) -> None:
        """Desenha a caixa da tela"""
        x1, y1, x2, y2 = map(int, deteccao.xyxy[0])
        confianca = float(deteccao.conf)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        label = f"Tela ({confianca:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def _processar_estribo(self, frame, deteccao) -> bool:
        """
        Processa detecção de estribo
        
        Returns:
            True se anomalia detectada
        """
        x1, y1, x2, y2 = map(int, deteccao.xyxy[0])
        confianca = float(deteccao.conf)
        diag("Processando estribo bbox=(%s,%s,%s,%s) conf=%.2f", x1, y1, x2, y2, confianca)
        
        erros = self.analisador.verificar_anomalias(x1, y1, x2, y2)
        anomalia = len(erros) > 0
        
        # Escolher cor
        cor = (0, 0, 255) if anomalia else (0, 255, 0)  # Vermelho ou Verde
        
        # Desenhar
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        label = f"Estribo ({confianca:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        
        # Registrar erros
        if anomalia:
            for erro in erros:
                logger.warning(f"⚠️ ANOMALIA: {erro}")
                diag("Anomalia detectada: %s", erro)
        
        return anomalia
    
    def executar(self) -> None:
        """Loop principal de monitoramento"""
        logger.info("Sistema de verificação de Estribo INICIADO")
        logger.info(f"Classes disponíveis: {self.modelo.names}")
        diag("Sistema iniciado, classes: %s", self.modelo.names)
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Erro na leitura da câmera")
                    diag("Falha ao ler frame da câmera")
                    break
                else:
                    diag("Frame lido com sucesso")
                
                # Processar frame
                tela, estribo, anomalia = self.processar_frame(frame)
                diag("Resultado processar_frame -> tela=%s estribo=%s anomalia=%s", tela, estribo, anomalia)
                
                # Controlar relé
                if anomalia:
                    diag("Anomalia detectada no frame: acionando rele")
                    self.rele.ligar()
                else:
                    diag("Nenhuma anomalia: garantindo relé desligado")
                    self.rele.desligar()
                
                # Timeout do relé
                self.rele.desligar_apos_timeout(self.tempo_rele_ativo)
                
                # Exibir resultado
                cv2.imshow("Monitoramento de Estribos", frame)
                
                # Tecla 'p' para sair
                if cv2.waitKey(1) & 0xFF == ord("p"):
                    logger.info("Encerrando...")
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuário")
        
        finally:
            self.limpar()
    
    def limpar(self) -> None:
        """Libera recursos"""
        self.rele.fechar()
        self.camera.release()
        cv2.destroyAllWindows()
        logger.info("Recursos liberados")
        diag("MonitorEstribo: recursos liberados")


def main():
    """Função principal"""
    try:
        monitor = MonitorEstribo(
            modelo_path=r"C:\Users\yan.fernandes\Downloads\weights.pt",
            porta_rele='COM5'
        )
        monitor.executar()
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())