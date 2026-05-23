# b3_sector_mask.py

import torch


B3_TICKERS = [
    "ABEV3", "AZZA3", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3",
    "BEEF3", "BOVA11", "BRAP4", "BRFS3", "BRKM5", "CMIG4", "COGN3",
    "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CVCB3", "CYRE3", "DIRR3",
    "EGIE3", "ELET3", "ELET6", "EMBR3", "ENEV3", "ENGI11", "EQTL3",
    "FLRY3", "GGBR4", "GOAU4", "HYPE3", "IGTI11", "ISAE4", "ITSA4",
    "ITUB4", "JBSS3", "KLBN11", "LREN3", "MGLU3", "MRFG3", "MRVE3",
    "MULT3", "NTCO3", "PCAR3", "PETR3", "PETR4", "POMO4", "PSSA3",
    "RADL3", "RENT3", "SANB11", "SBSP3", "SLCE3", "SMTO3", "STBP3",
    "SUZB3", "TAEE11", "TIMS3", "TOTS3", "UGPA3", "USIM5", "VALE3",
    "VIVT3", "WEGE3", "YDUQ3",
]


B3_SECTOR_LABELS = [
    "Consumo não cíclico", "Consumo cíclico", "Financeiro", "Financeiro",
    "Financeiro", "Financeiro", "Financeiro", "Consumo não cíclico",
    "ETF / índice", "Materiais básicos", "Consumo não cíclico",
    "Materiais básicos", "Utilidade pública", "Consumo cíclico",
    "Utilidade pública", "Utilidade pública", "Energia", "Materiais básicos",
    "Consumo cíclico", "Consumo cíclico", "Consumo cíclico",
    "Utilidade pública", "Utilidade pública", "Utilidade pública",
    "Industrial", "Utilidade pública", "Utilidade pública", "Utilidade pública",
    "Saúde", "Materiais básicos", "Materiais básicos", "Saúde",
    "Consumo cíclico", "Utilidade pública", "Financeiro", "Financeiro",
    "Consumo não cíclico", "Materiais básicos", "Consumo cíclico",
    "Consumo cíclico", "Consumo não cíclico", "Consumo cíclico",
    "Consumo cíclico", "Consumo não cíclico", "Consumo não cíclico",
    "Energia", "Energia", "Industrial", "Financeiro", "Saúde",
    "Consumo cíclico", "Financeiro", "Utilidade pública",
    "Consumo não cíclico", "Consumo não cíclico", "Industrial",
    "Materiais básicos", "Utilidade pública", "Comunicação", "Tecnologia",
    "Energia", "Materiais básicos", "Materiais básicos", "Comunicação",
    "Industrial", "Consumo cíclico",
]


def get_b3_sector_ids(device=None) -> torch.Tensor:
    """
    Retorna vetor (N,) com o id setorial de cada papel,
    na ordem fixa de B3_TICKERS.
    """
    sector_vocab = {
        sector: idx
        for idx, sector in enumerate(sorted(set(B3_SECTOR_LABELS)))
    }

    sector_ids = [sector_vocab[s] for s in B3_SECTOR_LABELS]

    return torch.tensor(sector_ids, dtype=torch.long, device=device)


def build_b3_cross_attention_mask(
    n_channels: int,
    seq_len: int,
    device=None,
    block_same_channel: bool = True,
    block_different_sector: bool = True,
    etf_sector_label: str = "ETF / índice",
) -> torch.Tensor:
    """
    Cria máscara booleana para MultiheadAttention.

    Shape:
        (N*T, N*T)

    Convenção PyTorch:
        True  = bloqueia atenção
        False = permite atenção

    Regra:
    - papéis comuns olham para papéis do mesmo setor;
    - papéis comuns também podem olhar para ETF / índice;
    - papéis comuns não olham para si mesmos, se block_same_channel=True;
    - ETF / índice, isto é, BOVA11, olha para todos os setores.
    """
    expected_n = len(B3_TICKERS)

    if n_channels != expected_n:
        raise ValueError(
            f"Máscara B3 espera {expected_n} canais, mas recebeu {n_channels}."
        )

    sector_vocab = {
        sector: idx
        for idx, sector in enumerate(sorted(set(B3_SECTOR_LABELS)))
    }

    if etf_sector_label not in sector_vocab:
        raise ValueError(
            f"Setor ETF não encontrado: {etf_sector_label}. "
            f"Setores válidos: {sorted(sector_vocab)}"
        )

    sector_ids = get_b3_sector_ids(device=device)
    etf_sector_id = sector_vocab[etf_sector_label]

    idx = torch.arange(n_channels * seq_len, device=device)
    ch = idx // seq_len

    token_sector = sector_ids[ch]

    mask = torch.zeros(
        n_channels * seq_len,
        n_channels * seq_len,
        dtype=torch.bool,
        device=device,
    )

    if block_different_sector:
        different_sector = token_sector[:, None] != token_sector[None, :]
        key_is_etf = token_sector[None, :] == etf_sector_id

        # Bloqueia setor diferente, exceto quando a chave é ETF/índice.
        mask |= different_sector & (~key_is_etf)

    if block_same_channel:
        mask |= ch[:, None] == ch[None, :]

    # BOVA11 / ETF pode olhar para todos os setores.
    # Isso também evita linha totalmente mascarada para o próprio BOVA11.
    query_is_etf = token_sector[:, None] == etf_sector_id
    mask = mask & (~query_is_etf)

    return mask