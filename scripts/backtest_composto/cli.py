import argparse
import sys
import traceback

from .pipeline import prepare, run_tasks


def parser():
    p = argparse.ArgumentParser(description="IC e backtests CPU: experimento composto, abertura-fechamento.")
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Validar insumos e congelar uma revisao do plano.")
    prep.add_argument("--predictions", required=True)
    prep.add_argument("--market", required=True)
    prep.add_argument("--returns-file", required=True)
    prep.add_argument("--strategies", required=True)
    prep.add_argument("--run-dir", required=True)
    prep.add_argument("--train", type=int, nargs="+", default=[80])
    prep.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 15, 20, 25])
    prep.add_argument("--slippage-bps", type=float, default=0.)
    prep.add_argument("--cost-bps", type=float, default=0.)
    prep.add_argument("--dry-run", action="store_true")
    prep.add_argument("--allow-local", action="store_true", help="Somente fixtures/testes fora do cluster.")
    run = commands.add_parser("run", help="Executar um lote de uma revisao congelada.")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--plan-file", help="Plano imutavel em planos/<signature>.json; padrao: plano atual.")
    run.add_argument("--workers", type=int, default=16)
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--shards", type=int, default=1)
    run.add_argument("--pilot", action="store_true", help="Executar apenas as tarefas piloto do composto.")
    merge = commands.add_parser("consolidate", help="Reunir resultados existentes; exit=1 se execucao incompleta.")
    merge.add_argument("--run-dir", required=True)
    return p


def main():
    args = parser().parse_args()
    try:
        status = prepare(args) if args.command == "prepare" else run_tasks(args)
    except (ValueError, OSError, KeyError) as error:
        if isinstance(error, OSError):
            traceback.print_exc(file=sys.stderr)
        print(f"ERRO: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    raise SystemExit(status or 0)
