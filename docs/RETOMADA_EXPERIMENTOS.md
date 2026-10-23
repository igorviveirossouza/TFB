# Retomada das grades compostas anteriores ao MASTER

A preparação original gravou uma assinatura que inclui `scripts/b3_experiment.py`.
A integração do MASTER alterou esse arquivo, de modo que os workers antigos
recusaram o código atualizado antes de treinar. A retomada mantém essa proteção:
recupera o código do commit registrado, verifica a assinatura original e usa uma
cópia isolada para os novos workers. O checkout de desenvolvimento pode continuar
recebendo mudanças do MASTER.

## Primeiro, auditar os resultados existentes

No checkout atualizado do cluster:

```bash
cd /sonic_home/igor.viveiros/src/TFB
bash scripts/recover_composite_experiments.sh audit
```

O launcher apenas copia os auxiliares e submete um coordenador a `gorgonas_dev`.
Nenhum Python, leitura dos Parquets ou treinamento acontece na phocus4. Ele imprime
o identificador do job e o diretório que receberá `report.json`. Guarde esse
relatório e os logs `coordenador_JOB.out` e `coordenador_JOB.err`.

Por padrão, a auditoria cobre estas quatro grades:

```text
comp_lb32-128-256-512_gaussian_0.025_complementary_k1_t80_v5
comp_lb32-128-256-512_uniform_0.025_complementary_k1_t80_v5
comp_lb32-128-256-512_cauchy_0.025_complementary_k1_t80_v5
comp_lb32-128-256-512_cauchy_0.025_natural_k1_t80_v5
```

A configuração vem dos `run_config.json` existentes. O comando original fornecido
pelo usuário identifica as grades, mas não substitui os valores efetivamente
registrados na preparação. Não se chama novamente o launcher antigo para preparar
ou sobrescrever esses manifestos.

## Critérios de reaproveitamento

- Uma tarefa concluída, com registro, assinatura e Parquet válidos, fica fora dos
  novos arrays. Ela não ocupa um job apenas para ser ignorada depois.
- Um resultado existente sem registro de conclusão, um hash divergente ou um
  artefato bruto que possa conter previsões fica bloqueado para investigação.
  A retomada não apaga esse arquivo nem o trata automaticamente como treino faltante.
- Somente tarefas classificadas como ausentes podem ser submetidas.
- Se o commit original não reproduzir a assinatura registrada, a retomada para.
  Isso pode ocorrer quando havia alterações locais não commitadas na preparação;
  não se substitui a assinatura para contornar a divergência.

Previsões persistidas podem ser reaproveitadas. Um processo cancelado antes de
salvar previsões ou um checkpoint utilizável não oferece estado suficiente para
retomar exatamente do ponto interrompido. Os relatórios distinguem o que está
concluído, ausente e bloqueado; não há promessa de recuperar estado que não foi salvo.

## Submeter as tarefas ausentes

A ação abaixo também pode ser executada diretamente: ela audita antes de submeter
e não precisa de uma aprovação manual intermediária. Se a auditoria anterior
apontou bloqueios, resolva-os antes de tentar a retomada.

```bash
bash scripts/recover_composite_experiments.sh resume
```

Essa ação realiza nova auditoria das grades selecionadas antes de submeter. Se
qualquer grade apresentar bloqueio, nenhum worker novo será submetido. Os workers executam o código
isolado e a configuração original, com mapeamento explícito dos índices globais
das tarefas. As saídas continuam nos diretórios originais, preservando os resultados
já concluídos.

Bloqueios encontrados na auditoria impedem todos os novos workers. Se o Slurm
aceitar um array e falhar durante uma submissão posterior, o relatório recebe
`SUBMISSION_INCOMPLETE` e mostra os registros de submissão, incluindo os IDs já
aceitos. Nesse caso, não se presume que a fila esteja vazia. O diário
`.b3_recovery_submissions.json`, em cada diretório de grade, impede duplicações;
uma resposta ambígua do Slurm exige conferir o token registrado antes de repetir.

O relatório `report.json` e os dois logs do coordenador bastam para a primeira
revisão da execução no cluster. Os manifestos originais também são copiados para
o diretório dessa solicitação; as bases e os Parquets permanecem onde estão.

Os recursos padrão dos workers são `gorgonas`, 1h30, 3 CPUs e no máximo 2
tarefas simultâneas por array; GRES e lista de nós ficam vazios. A memória não
recebe um limite adicional: vale o padrão da partição, como na submissão original
sem `MEMORY`. As variáveis antigas
`PARTITION`, `GRES` e `NODELIST` não redirecionam essa retomada. Para ajustar apenas
recursos, use os nomes próprios abaixo, sem alterar a configuração científica:

```bash
RECOVERY_WORKER_TIME=08:00:00 RECOVERY_WORKER_MEMORY=16G \
  bash scripts/recover_composite_experiments.sh resume
```

| Variável | Padrão |
|---|---|
| `RECOVERY_ROOT` | `/snfs2/igor.viveiros/retomadas_tfb` |
| `RECOVERY_OUT_BASE` | `/snfs2/igor.viveiros/previsoes` |
| `RECOVERY_RESULT_BASE` | `/snfs2/igor.viveiros/result` |
| `RECOVERY_WORKER_PARTITION` | `gorgonas` |
| `RECOVERY_WORKER_TIME` | `01:30:00` |
| `RECOVERY_WORKER_CPUS` | `3` |
| `RECOVERY_WORKER_MEMORY` | vazio; usa o padrão da partição |
| `RECOVERY_WORKER_GRES` | vazio |
| `RECOVERY_WORKER_NODELIST` | vazio |
| `RECOVERY_MAX_CONCURRENT` | `2` |
| `RECOVERY_AUDIT_PARTITION` | `gorgonas_dev` |
| `RECOVERY_AUDIT_TIME` | `01:30:00` |
| `RECOVERY_AUDIT_CPUS` | `3` |
| `RECOVERY_AUDIT_MEMORY` | `8G` |
| `PYTHON_BIN` | `/sonic_home/igor.viveiros/py310/bin/python` |

Para selecionar apenas uma grade, acrescente `--run-tag NOME`; a opção pode ser
repetida. Os diretórios de recuperação são novos a cada solicitação. Não faça
`git pull` nem edite código dentro das cópias isoladas: mantenha o desenvolvimento
no checkout original.
