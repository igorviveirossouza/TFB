#!/usr/bin/env Rscript
# Experimento isolado: NS, treino80, LB256/512, H25, K_loss1/5/25.
# Nao altera insumos, nao executa/source() pipelines antigos e nao treina modelos.
# 38 ajustes: 2 MSE (lambda0) + 2 LB * 3 lambdas * 3 K_loss * 2 losses.
# Parquets esperados: retornos LOG economicos JA despadronizados pelo forecast.
# NAO aplicar inverse_transform, exp, diferenca de precos ou pooling de datas.
# IC = Spearman transversal por origem; resumo = mediana desses ICs diarios.
# Use --help e --self-test sem pacotes externos. Execucao real requer:
# install.packages(c("arrow", "jsonlite", "ggplot2", "digest"))
# Exemplo (caminhos locais depois da transferencia do cluster):
# Rscript analise_ic_ns_kloss.R --pred-root /dados/ns_kloss_isolado_t80_20261105 \
#   --returns /dados/b3_daily_tfb_new.csv --calendar /dados/calendario_tfb.csv \
#   --out /dados/analise_ns_kloss_20261105

fail <- function(...) stop(..., call. = FALSE)
need <- function(x, ...) if (!isTRUE(x)) fail(...)
safe_ic <- function(x, y) {
  need(length(x) == length(y), "Vetores IC com universos diferentes.")
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3L || length(unique(x[ok])) < 2L ||
      length(unique(y[ok])) < 2L) return(NA_real_)
  unname(stats::cor(x[ok], y[ok], method = "spearman"))
}
ic_rows <- function(x, y) {
  need(identical(dim(x), dim(y)), "Matrizes IC com dimensoes diferentes.")
  vapply(seq_len(nrow(x)), function(i) safe_ic(x[i, ], y[i, ]), numeric(1))
}
prefix_double <- function(x) {
  # Soma explicita em double; o indice1 representa o prefixo vazio.
  out <- matrix(0, nrow(x) + 1L, ncol(x))
  for (i in seq_len(nrow(x))) out[i + 1L, ] <- out[i, ] + x[i, ]
  out
}
future_sum <- function(x, origins, k) {
  need(all(origins >= 1L & origins + k <= nrow(x)), "Alvo fora do painel.")
  out <- matrix(0, length(origins), ncol(x))
  for (h in seq_len(k)) out <- out + x[origins + h, , drop = FALSE]
  out
}
momentum_score <- function(prefix, origins, lb) {
  need(all(origins >= lb), "Lookback sem historico completo.")
  prefix[origins + 1L, , drop = FALSE] - prefix[origins - lb + 1L, , drop = FALSE]
}
validate_prediction <- function(d, assets, origins, horizon) {
  need(all(c("h", "origin_step", "step", assets) %in% names(d)),
       "Parquet sem chaves h/origin_step/step ou sem universo completo.")
  extras <- setdiff(names(d), c(assets, "h", "origin_step", "step", "arquivo",
    "dataset", "modelo", "lookback", "pred_len", "k_dir", "treino", "experimento",
    "loss", "lambda", "delta", "mask_type", "mask_a", "mask_mode", "seed"))
  need(length(extras) == 0L, "Colunas inesperadas no parquet: ", paste(extras, collapse = ", "))
  for (nm in c("h", "origin_step", "step")) {
    need(is.numeric(d[[nm]]) && all(is.finite(d[[nm]])) &&
         all(d[[nm]] == floor(d[[nm]])), "Chave nao inteira: ", nm)
    d[[nm]] <- as.integer(d[[nm]])
  }
  need(!anyDuplicated(d[c("origin_step", "h")]), "Duplicatas de origem/h no parquet.")
  need(all(d$step - d$origin_step == d$h), "Desalinhamento: step-origin_step != h.")
  need(identical(sort(unique(d$origin_step)), as.integer(origins)),
       "Origens diferentes da amostra comum fixa; nao sera feita intersecao silenciosa.")
  need(nrow(d) == length(origins) * horizon && all(d$h %in% seq_len(horizon)),
       "Previsoes incompletas: cada origem deve conter exatamente h=1..H.")
  d <- d[order(d$origin_step, d$h), , drop = FALSE]
  need(identical(d$h, rep(seq_len(horizon), length(origins))), "Horizontes incompletos.")
  need(all(vapply(d[assets], is.numeric, logical(1))), "Previsoes nao numericas.")
  x <- as.matrix(d[assets])
  need(all(is.finite(x)), "Previsoes nao finitas; universo comum de 55 ativos nao preservado.")
  x
}
prediction_scores <- function(x, n_origins, horizon, ks) {
  running <- matrix(0, n_origins, ncol(x))
  out <- setNames(vector("list", length(ks)), as.character(ks))
  for (h in seq_len(horizon)) {
    running <- running + x[seq.int(h, nrow(x), by = horizon), , drop = FALSE]
    if (h %in% ks) out[[as.character(h)]] <- running
  }
  out
}
self_test <- function() {
  x <- matrix(seq_len(48) / 100, nrow = 12, ncol = 4)
  origins <- 5:8
  p <- prefix_double(x)
  need(isTRUE(all.equal(future_sum(x, origins, 2), x[origins + 1, ] + x[origins + 2, ])),
       "Self-test: janela futura incorreta.")
  need(isTRUE(all.equal(momentum_score(p, origins, 3),
       x[origins, ] + x[origins - 1, ] + x[origins - 2, ])), "Self-test: momentum incorreto.")
  need(safe_ic(1:4, 1:4) == 1 && safe_ic(1:4, 4:1) == -1,
       "Self-test: Spearman incorreto.")
  need(is.na(safe_ic(rep(1, 4), 1:4)), "Self-test: vetor constante precisa produzir NA.")
  assets <- paste0("A", 1:4)
  d <- data.frame(h = rep(1:2, 4), origin_step = rep(origins, each = 2))
  d$step <- d$origin_step + d$h
  d[assets] <- as.data.frame(matrix(seq_len(32) / 100, nrow = 8, ncol = 4))
  v <- validate_prediction(d, assets, origins, 2)
  s <- prediction_scores(v, 4, 2, c(1, 2))
  need(isTRUE(all.equal(s[["2"]], v[c(1, 3, 5, 7), ] + v[c(2, 4, 6, 8), ])),
       "Self-test: cumulacao de horizontes incorreta.")
  broken <- d; broken$step[1] <- broken$step[1] + 1L
  need(inherits(try(validate_prediction(broken, assets, origins, 2), silent = TRUE), "try-error"),
       "Self-test: alinhamento invalido deveria falhar.")
  need(inherits(try(validate_prediction(d[-1, ], assets, origins, 2), silent = TRUE), "try-error"),
       "Self-test: horizonte ausente deveria falhar.")
  # Mediana por data difere de uma correlacao sobre observacoes empilhadas.
  a <- rbind(1:4, 101:104, 201:204); b <- rbind(4:1, 104:101, 204:201)
  need(stats::median(ic_rows(a, b)) == -1 && safe_ic(as.vector(a), as.vector(b)) > 0,
       "Self-test: mediana diaria confundida com pooling.")
  cat("SELF-TEST OK: janelas futuras, momentum, Spearman, horizontes, origem e mediana diaria.\n")
}
usage <- function() cat(paste0(
  "Uso: Rscript analise_ic_ns_kloss.R --pred-root DIRETORIO --returns CSV_LOG_RETORNOS ",
  "--out DIRETORIO_NOVO [--calendar CSV] [--seed 2026] [--allow-partial]\n",
  "     Rscript analise_ic_ns_kloss.R --self-test | --help\n",
  "Grade fixa: NS, treino = 80%, LB = 256/512, H = 25, lambda = 0/.5/.8/1, K_loss = 1/5/25.\n",
  "ROOT inclui subpastas temporal e composto, run_config.json e part.parquet.\n",
  "CSV de retornos LOG corrigidos: date (indice 1..T), cols (ticker), data (retorno).\n",
  "O padrao exige 38 ajustes e 709 origens identicas. --allow-partial permite SOMENTE\n",
  "configuracoes ausentes: nunca datas/ativos/horizontes ausentes. A figura sera rotulada incompleta.\n",
  "Nao consome resumos antigos nem transforma preco bruto. Nao inventa resultados novos.\n"
))

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if ("--help" %in% args || !length(args)) { usage(); return(invisible(NULL)) }
  if (identical(args, "--self-test")) { self_test(); return(invisible(NULL)) }
  allowed <- c("--pred-root", "--returns", "--calendar", "--out", "--seed", "--allow-partial")
  opt <- list(seed = "2026", allow_partial = FALSE)
  i <- 1L
  while (i <= length(args)) {
    flag <- args[i]; need(flag %in% allowed, "Opcao desconhecida: ", flag)
    if (flag == "--allow-partial") { opt$allow_partial <- TRUE; i <- i + 1L; next }
    need(i < length(args) && !startsWith(args[i + 1L], "--"), "Valor ausente para ", flag)
    opt[[gsub("-", "_", substring(flag, 3L))]] <- args[i + 1L]; i <- i + 2L
  }
  for (nm in c("pred_root", "returns", "out")) need(!is.null(opt[[nm]]), "Opcao obrigatoria: --", gsub("_", "-", nm))
  packages <- c("arrow", "jsonlite", "ggplot2", "digest")
  absent <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  need(!length(absent), "Instale os pacotes R: ", paste(absent, collapse = ", "))
  need(dir.exists(opt$pred_root), "Pasta de previsoes nao encontrada.")
  need(file.exists(opt$returns), "CSV de log-retornos nao encontrado.")
  root <- normalizePath(opt$pred_root, winslash = "/", mustWork = TRUE)
  out <- normalizePath(opt$out, winslash = "/", mustWork = FALSE)
  need(!(identical(root, out) || startsWith(out, paste0(root, "/"))), "Saida deve ficar fora da pasta de previsoes.")
  need(!dir.exists(out) || !length(list.files(out, all.files = TRUE, no.. = TRUE)), "Saida ja contem arquivos; escolha novo --out.")
  seed <- suppressWarnings(as.integer(opt$seed)); need(!is.na(seed), "Seed invalida.")
  lbs <- c(256L, 512L); ks <- c(1L, 5L, 10L, 15L, 20L, 25L)
  lambdas <- c(0, 0.5, 0.8, 1); kloss <- c(1L, 5L, 25L); H <- 25L
  model <- "Nonstationary_Transformer"
  src_hash <- digest::digest(opt$returns, algo = "sha256", file = TRUE, serialize = FALSE)
  raw <- utils::read.csv(opt$returns, check.names = FALSE, stringsAsFactors = FALSE)
  need(identical(sort(names(raw)), sort(c("date", "data", "cols"))),
       "CSV deve conter exatamente date,data,cols; use b3_daily_tfb_new.csv de log-retornos.")
  need(is.numeric(raw$date) && all(is.finite(raw$date)) && all(raw$date == floor(raw$date)), "date deve ser indice inteiro 1..T.")
  need(is.numeric(raw$data) && all(is.finite(raw$data)), "Retornos devem ser numericos e finitos.")
  need(!anyDuplicated(raw[c("date", "cols")]), "Retornos duplicados por data/ticker.")
  assets <- sort(unique(raw$cols)); dates <- sort(unique(as.integer(raw$date)))
  need(length(assets) == 55L && identical(dates, seq_len(3664L)) && nrow(raw) == 3664L * 55L,
       "Este experimento exige painel 3664 x 55 completo; nao reduzir universo ou datas silenciosamente.")
  raw <- raw[order(raw$date, raw$cols), ]
  returns <- matrix(raw$data, nrow = length(dates), ncol = length(assets), byrow = TRUE)
  colnames(returns) <- assets
  origins <- seq.int(floor(length(dates) * 0.8), length(dates) - H)
  need(identical(origins, 2931L:3639L), "Corte 80% / H = 25 inesperado.")
  origin_dates <- rep(NA_character_, length(origins))
  if (!is.null(opt$calendar)) {
    cal <- utils::read.csv(opt$calendar, check.names = FALSE, stringsAsFactors = FALSE)
    step_col <- intersect(c("step", "date", "index", "indice"), names(cal))
    day_col <- intersect(c("date_pregao", "data_pregao", "data"), names(cal))
    need(length(step_col) == 1L && length(day_col) == 1L, "Calendario precisa step/date e date_pregao/data_pregao.")
    cal_steps <- as.integer(cal[[step_col]]); cal_dates <- as.Date(cal[[day_col]])
    need(!anyNA(cal_steps) && !anyNA(cal_dates) && !anyDuplicated(cal_steps) &&
         setequal(cal_steps, dates), "Calendario nao cobre exatamente os 3664 indices.")
    cal_dates <- cal_dates[match(dates, cal_steps)]
    need(all(diff(cal_dates) > 0), "Calendario fora da ordem cronologica.")
    origin_dates <- as.character(cal_dates[origins])
  }
  files <- list.files(root, pattern = "^part[.]parquet$", recursive = TRUE, full.names = TRUE)
  need(length(files) > 0L, "Nenhum part.parquet encontrado; nenhuma analise foi produzida.")
  token <- function(path, key) {
    parts <- strsplit(gsub("\\\\", "/", path), "/", fixed = TRUE)[[1L]]
    vals <- parts[startsWith(parts, paste0(key, "="))]
    if (!length(vals)) return(NA_character_)
    need(length(vals) == 1L, "Metadado repetido no caminho: ", key, " em ", path)
    substring(vals, nchar(key) + 2L)
  }
  number <- function(x) suppressWarnings(as.numeric(x))
  config_cache <- new.env(parent = emptyenv())
  get_config <- function(path) {
    here <- dirname(path)
    repeat {
      cfg_path <- file.path(here, "run_config.json")
      if (file.exists(cfg_path)) break
      need(!identical(here, root) && !identical(dirname(here), here), "run_config.json ancestral ausente: ", path)
      here <- dirname(here)
    }
    if (!exists(cfg_path, config_cache, inherits = FALSE)) {
      cfg <- jsonlite::fromJSON(cfg_path, simplifyVector = FALSE)
      need(identical(cfg$metadata$source_sha256, src_hash), "Hash dos retornos diverge do treinamento: ", cfg_path)
      need(identical(cfg$metadata$source_data_kind, "log_return") &&
           identical(cfg$config$source_data_kind, "log_return"), "Treino nao declara fonte log_return.")
      need(as.integer(cfg$metadata$n_assets) == 55L && as.integer(cfg$metadata$n_rows) == 3664L &&
           setequal(unlist(cfg$metadata$targets), assets), "Universo/calendario do treino diverge da fonte.")
      need(as.integer(cfg$metadata$step_offset) == 1L, "step_offset do treinamento deve ser 1.")
      need(as.integer(cfg$config$strategy$seed) == seed, "Seed diferente da solicitada.")
      need(isTRUE(cfg$config$loss$loss_inverse_norm), "Config nao confirma loss_inverse_norm=TRUE; verificar contrato de escala.")
      need(identical(cfg$config$loss$loss_temporal, "mse"), "Controle temporal precisa ser MSE.")
      need(identical(cfg$config$loss$loss_data_kind, "log_return") &&
           identical(cfg$config$loss$loss_score_kind, "simple_return") &&
           identical(cfg$config$loss$loss_cross_score_normalization, "zscore"),
           "Escalas da loss devem ser log_return/simple_return/zscore.")
      need(identical(as.numeric(cfg$config$loss$loss_cross_scale), 1) &&
           identical(as.numeric(cfg$config$loss$loss_ranknet_alpha), 1),
           "Este desenho exige cross_scale = 1 e ranknet_alpha = 1.")
      need(as.integer(cfg$config$model_defaults[[model]]$batch_size) == 16L,
           "Este desenho exige batch_size = 16 em todos os treinamentos.")
      cfg$.path <- cfg_path
      assign(cfg_path, cfg, envir = config_cache)
    }
    get(cfg_path, config_cache, inherits = FALSE)
  }
  configs <- list()
  for (f in files) {
    tr <- number(token(f, "treino")); lb <- number(token(f, "lookback")); hh <- number(token(f, "pred_len"))
    if (is.na(tr) || is.na(lb) || is.na(hh) || tr != 80 || !(lb %in% lbs) || hh != H ||
        !identical(token(f, "modelo"), model)) next
    exp <- token(f, "experimento"); loss <- token(f, "loss"); lam <- number(token(f, "lambda"))
    ktrain <- number(token(f, "k_dir"))
    if (identical(exp, "temporal")) {
      need(identical(loss, "mse") && is.finite(lam) && lam == 0,
           "Ramo temporal deve declarar loss=mse/lambda=00 no caminho: ", f)
      if (is.na(ktrain)) ktrain <- 1
    }
    if (is.na(lam) || !(lam %in% lambdas) || is.na(ktrain) || !(ktrain %in% kloss)) next
    if (lam == 0 && !(ktrain == 1 && (identical(exp, "temporal") || identical(loss, "ranknet")))) next
    if (lam > 0 && !(loss %in% c("ranknet", "ranknet_proximity"))) next
    cfg <- get_config(f)
    need(80 %in% unlist(cfg$config$train_percentages) && lb %in% unlist(cfg$config$lookbacks) &&
         H %in% unlist(cfg$config$horizons) && model %in% unlist(cfg$config$models) &&
         ktrain %in% unlist(cfg$config$trade_windows),
         "Metadados do caminho nao pertencem a grade declarada no run_config: ", f)
    if (lam > 0) need(lam %in% unlist(cfg$config$lambdas) && loss %in% unlist(cfg$config$cross_losses) &&
         0 %in% unlist(cfg$config$deltas), "Loss/lambda/delta do caminho nao pertencem ao run_config: ", f)
    delta <- number(token(f, "delta")); if (!is.na(delta)) need(delta == 0, "Delta nao zero: ", f)
    variant <- if (lam == 0) "Temporal MSE" else if (loss == "ranknet") "RankNet" else "Cauchy natural"
    if (variant == "Cauchy natural") {
      mask <- unlist(cfg$config$loss$loss_cross_mask, use.names = FALSE)
      need(length(mask) %in% c(2L, 3L) && mask[1] == "cauchy" && number(mask[2]) == 0.025 &&
           (length(mask) == 2L || mask[3] == "natural"), "Proximity exige Cauchy natural a = 0.025.")
      need(identical(token(f, "mask"), "cauchy") && identical(number(token(f, "a")), 0.025),
           "Caminho de Proximity deve declarar mask=cauchy/a=0.025: ", f)
      path_mode <- c(token(f, "mask_mode"), token(f, "mode"))
      path_mode <- path_mode[!is.na(path_mode)]
      need(!length(path_mode) || all(path_mode == "natural"), "Modo da mascara no caminho diverge de natural: ", f)
      # O runner omite o segmento de modo no formato natural canonico de2 elementos.
    }
    need(H %% ktrain == 0L, "K_loss nao divide H.")
    configs[[length(configs) + 1L]] <- data.frame(file = f, lookback = lb, lambda = lam,
      k_loss = ktrain, variant = variant, config_path = cfg$.path, stringsAsFactors = FALSE)
  }
  need(length(configs) > 0L, "Nenhuma configuracao elegivel da grade nova.")
  design <- do.call(rbind, configs)
  key <- function(d) paste(d$lookback, d$lambda, d$k_loss, d$variant, sep = "|")
  need(!anyDuplicated(key(design)), "Configuracoes duplicadas (incluindo controles lambda = 0); nao agregar arquivos de runs diferentes.")
  expected <- expand.grid(lookback = lbs, lambda = lambdas[-1], k_loss = kloss,
    variant = c("RankNet", "Cauchy natural"), stringsAsFactors = FALSE)
  expected <- rbind(expected, data.frame(lookback = lbs, lambda = 0, k_loss = 1, variant = "Temporal MSE"))
  missing <- expected[!(key(expected) %in% key(design)), , drop = FALSE]
  need(opt$allow_partial || nrow(missing) == 0L, "Grade incompleta: faltam ", nrow(missing),
       " de 38 ajustes. Use --allow-partial somente para inspecao explicitamente incompleta.")
  # Todas as configuracoes precisam compartilhar os mesmos defaults de arquitetura e treino.
  unique_cfg <- lapply(unique(design$config_path), function(p) get(p, config_cache, inherits = FALSE))
  invariant <- vapply(unique_cfg, function(c) jsonlite::toJSON(list(
    model = c$config$model_defaults[[model]], training = c$config$training,
    strategy = c$config$strategy), auto_unbox = TRUE), character(1))
  need(length(unique(invariant)) == 1L, "Defaults de arquitetura/batch/treino/seed diferem entre os runs.")
  target <- setNames(lapply(ks, function(k) future_sum(returns, origins, k)), as.character(ks))
  prefix <- prefix_double(returns)
  daily <- vector("list", nrow(design) + length(lbs))
  for (i in seq_len(nrow(design))) {
    cfg <- design[i, ]; message("IC ", i, "/", nrow(design), ": LB = ", cfg$lookback,
      " lambda = ", cfg$lambda, " K_loss = ", cfg$k_loss, " ", cfg$variant)
    p <- as.data.frame(arrow::read_parquet(cfg$file))
    x <- validate_prediction(p, assets, origins, H)
    scores <- prediction_scores(x, length(origins), H, ks)
    daily[[i]] <- do.call(rbind, lapply(ks, function(k) data.frame(
      lookback = cfg$lookback, lambda = cfg$lambda, k_loss = cfg$k_loss,
      variant = cfg$variant, k_eval = k, origin_step = origins,
      origin_date = origin_dates, spearman = ic_rows(scores[[as.character(k)]], target[[as.character(k)]]))))
    rm(p, x, scores)
  }
  for (j in seq_along(lbs)) {
    lb <- lbs[j]; scores <- momentum_score(prefix, origins, lb)
    daily[[nrow(design) + j]] <- do.call(rbind, lapply(ks, function(k) data.frame(
      lookback = lb, lambda = NA_real_, k_loss = 0L, variant = "Momentum", k_eval = k,
      origin_step = origins, origin_date = origin_dates,
      spearman = ic_rows(scores, target[[as.character(k)]]))))
  }
  daily <- do.call(rbind, daily)
  need(all(is.finite(daily$spearman)), "IC invalido em alguma origem; nao reduzir amostra por configuracao.")
  daily$panel <- ifelse(daily$variant == "Momentum", "Momentum", paste0("lambda = ", daily$lambda))
  group_cols <- c("lookback", "panel", "lambda", "k_loss", "variant", "k_eval")
  groups <- split(seq_len(nrow(daily)), interaction(daily[c("lookback", "panel", "k_loss", "variant", "k_eval")], drop = TRUE))
  summary <- do.call(rbind, lapply(groups, function(ix) {
    g <- daily[ix, ]; need(nrow(g) == 709L && identical(sort(g$origin_step), origins), "Grupo nao possui 709 origens comuns.")
    cbind(g[1, group_cols, drop = FALSE], n_origens = nrow(g), n_ativos = 55L,
      ic_medio = mean(g$spearman), ic_mediano = median(g$spearman),
      ic_desvio = sd(g$spearman), amostra = "comum", treino = 80L, pred_len = H, seed = seed)
  }))
  reference_key <- function(d) paste(d$lookback, d$k_eval, sep = "|")
  for (reference in c("Momentum", "Temporal MSE")) {
    ref <- summary[summary$variant == reference, , drop = FALSE]
    suffix <- if (reference == "Momentum") "momentum" else "temporal"
    matched <- match(reference_key(summary), reference_key(ref))
    for (metric in c("ic_medio", "ic_mediano")) {
      summary[[paste0(metric, "_", suffix)]] <- ref[[metric]][matched]
      summary[[paste0("delta_", metric, "_vs_", suffix)]] <- summary[[metric]] - ref[[metric]][matched]
    }
  }
  panels <- c(paste0("lambda = ", lambdas), "Momentum")
  summary$panel <- factor(summary$panel, levels = panels)
  summary$lookback_panel <- factor(paste0("LB = ", summary$lookback), levels = paste0("LB = ", lbs))
  curves <- as.vector(outer(c("RankNet", "Cauchy natural"), kloss, function(v, k) paste0("K_loss = ", k, " | ", v)))
  summary$curve <- factor(paste0("K_loss = ", summary$k_loss, " | ", summary$variant), levels = curves)
  colors <- setNames(rep(c("#0072B2", "#D55E00", "#009E73"), each = 2L), curves)
  types <- setNames(rep(c("solid", "dashed"), length(kloss)), curves)
  plot_data <- summary[summary$variant %in% c("RankNet", "Cauchy natural"), ]
  zero <- summary[summary$variant == "Temporal MSE", ]; mom <- summary[summary$variant == "Momentum", ]
  frame <- expand.grid(lookback_panel = factor(paste0("LB = ", lbs), levels = paste0("LB = ", lbs)),
    panel = factor(panels, levels = panels), k_eval = ks)
  frame$ic_mediano <- NA_real_
  incomplete <- nrow(missing) > 0L
  p <- ggplot2::ggplot(frame, ggplot2::aes(x = k_eval, y = ic_mediano)) +
    ggplot2::geom_blank() + ggplot2::geom_hline(yintercept = 0, color = "grey75", linewidth = .3) +
    ggplot2::geom_line(data = plot_data, ggplot2::aes(color = curve, linetype = curve, group = curve), linewidth = .65) +
    ggplot2::geom_point(data = plot_data, ggplot2::aes(color = curve), size = 1.6) +
    ggplot2::geom_line(data = zero, color = "#222222", linewidth = .8) +
    ggplot2::geom_point(data = zero, color = "#222222", size = 1.6) +
    ggplot2::geom_line(data = mom, color = "#222222", linetype = "dashed", linewidth = .8) +
    ggplot2::geom_point(data = mom, color = "#222222", shape = 17, size = 1.7) +
    ggplot2::facet_grid(lookback_panel ~ panel, scales = "fixed", drop = FALSE) +
    ggplot2::scale_x_continuous(breaks = ks, limits = range(ks)) +
    ggplot2::scale_color_manual(values = colors, breaks = curves, drop = FALSE, name = "Loss cross e janela de treino") +
    ggplot2::scale_linetype_manual(values = types, breaks = curves, drop = FALSE, name = "Loss cross e janela de treino") +
    ggplot2::labs(title = paste0("Nonstationary Transformer | treino = 80% | H = 25", if (incomplete) " | GRADE INCOMPLETA" else ""),
      subtitle = "Mediana dos ICs transversais diarios | mesmas 709 origens e 55 ativos em todos os paineis",
      x = "Janela de avaliacao K_eval (pregoes)", y = "IC de Spearman mediano",
      caption = paste0("lambda = 0: controle MSE unico, sem K_loss ativo. Momentum: retorno historico do respectivo LB.\n",
        "Cauchy natural a = 0.025; seed = ", seed, ". K_eval nao e K_loss. ",
        if (incomplete) paste0(nrow(missing), " configuracoes ausentes; espacos vazios nao sao resultados.") else "38 ajustes completos.")) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(legend.position = "bottom", legend.box = "vertical", panel.grid.minor = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold"), axis.text.x = ggplot2::element_text(size = 8),
      plot.caption = ggplot2::element_text(hjust = 0)) +
    ggplot2::guides(color = ggplot2::guide_legend(nrow = 2, byrow = TRUE), linetype = ggplot2::guide_legend(nrow = 2, byrow = TRUE))
  # Escreve somente depois de validar todas as configuracoes e origens.
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(file.path(out, "ic_ns_kloss_2x5.pdf"), p, width = 16, height = 7.3, units = "in")
  ggplot2::ggsave(file.path(out, "ic_ns_kloss_2x5.png"), p, width = 16, height = 7.3, units = "in", dpi = 180)
  utils::write.csv(summary[setdiff(names(summary), c("curve", "lookback_panel"))], file.path(out, "ic_resumo.csv"), row.names = FALSE)
  arrow::write_parquet(daily, file.path(out, "ic_diario.parquet"))
  utils::write.csv(design, file.path(out, "fontes.csv"), row.names = FALSE)
  utils::write.csv(missing, file.path(out, "configuracoes_ausentes.csv"), row.names = FALSE)
  audit <- list(status = if (incomplete) "INCOMPLETE_EXPLICIT" else "COMPLETE", expected_models = 38L,
    observed_models = nrow(design), source_sha256 = src_hash, origins = list(first = min(origins), last = max(origins), n = length(origins)),
    assets = assets, seed = seed, actual_configs = unique(design$config_path),
    input_files_sha256 = setNames(lapply(design$file, function(f) digest::digest(f, algo = "sha256", file = TRUE, serialize = FALSE)), design$file),
    semantics = "Median of per-origin cross-sectional Spearman; predicted and realized cumulative economic log returns; no pooled correlations; no inverse scaling repeated.",
    parser = "R utils::read.csv numeric parser; prefix and forecast accumulation use explicit double additions.", session = capture.output(sessionInfo()))
  jsonlite::write_json(audit, file.path(out, "auditoria.json"), auto_unbox = TRUE, pretty = TRUE)
  message("Concluido: ", out)
  invisible(summary)
}

if (sys.nframe() == 0L) main()
