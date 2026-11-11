const REQUEST = `POST /api/predict
Content-Type: multipart/form-data

image        file    рентгенограмма (png / jpg / tif), до 64 МБ и 50 Мпикс;
                     16 бит окно по перцентилям, как в обучающих данных
mc_samples   int     число проходов MC Dropout T (2–30, по умолчанию 10)
max_dim      int     длинная сторона возвращаемых карт (64–4096, по умолчанию 1024)

GET /api/health      -> { status, model, stub, classes, device, mode, ... }
GET /api/samples     -> патчи датасета: сплит, образец, метки
GET /api/samples/<id>.png`;

const RESPONSE = `{
  "width": 1024, "height": 384,        // размер выхода: то же правило, что у canvas клиента
  "classes": ["porosity", "inclusion", "crack",
              "undercut", "lack_of_fusion",
              "lack_of_penetration"],
  "mask_png": "<base64>",              // серый PNG, значение = индекс класса, 0 = фон
  "maps": {                            // серый PNG 0–255 → 0–1, тот же размер
    "pe": "<base64>",                  // предсказательная энтропия / ln(K+1)
    "mi": "<base64>"                   // взаимная информация, ×4 для отображения
  },
  "defects": [{
    "cls": "crack",
    "prob": 0.91,                      // средняя вероятность дефекта в области
    "bbox": [x, y, w, h],              // в пикселях выхода
    "uncertainty": { "pe": 0.18, "mi": 0.05 }   // среднее по области
  }],
  "tiles": [{ "bbox": [...], "probs": {...}, "std": {...}, "pe": 0.4, "mi": 0.1 }],
                                       // только в режиме патчей (RIAWELC)
  "model": "swrd_baseline (epoch 12)", "stub": false, "mc_samples": 10
}`;

export function ApiView() {
  return (
    <>
      <div className="results-head">
        <p className="kicker">Интеграция</p>
        <h2>Один эндпоинт между интерфейсом и моделью.</h2>
        <p className="lede">Сервер на FastAPI — в <code>src/weldsight/serve/</code>. Инференс режет снимок на тайлы и делает MC Dropout так же, как на val/test (тайлы <code>patch_size</code>,
          нормализация по патчу), и возвращает ответ в формате ниже.</p>
      </div>
      <div className="split">
        <section className="block">
          <h3>Запуск</h3>
          <pre><code>{`# готовая модель этапа 1 (RIAWELC) из релиза
uv sync --extra serve
uv run weldsight-fetch-model
uv run weldsight-serve --run models/riawelc_cpu_v1 --static frontend/dist

# свой прогон (SWRD / GDXray — сегментация, RIAWELC — патчи)
uv run weldsight-serve --run runs/riawelc_cpu_20261111-053129 --static frontend/dist

# без модели: эвристическая заглушка
uv run weldsight-serve --stub

# режим разработки UI (прокси /api на :8000)
cd frontend && npm install && npm run dev`}</code></pre>
          <h3>Запрос</h3>
          <pre><code>{REQUEST}</code></pre>
        </section>
        <section className="block">
          <h3>Ответ <small>application/json</small></h3>
          <pre><code>{RESPONSE}</code></pre>
        </section>
      </div>
      <section className="block wide">
        <h3>Примечания</h3>
        <ul className="notes">
          <li>Индекс класса 0 — фон, дефекты 1..K в порядке <code>classes</code> (как в <code>data/sources.py</code>).</li>
          <li>Область считается <b>обнаружением</b>, если средняя вероятность дефекта ≥ 0.5. Более слабые (≥ 0.2 попиксельно) возвращаются как <b>под вопросом</b>, чтобы неуверенный пропуск всё равно можно было передать инспектору.</li>
          <li>Интерфейс ранжирует снимок по <b>максимальной</b> неуверенности области. Это аналог, но <b>не то же самое</b>, что оценка снимка в кривой передачи этапа 1, поэтому пороги с той кривой несопоставимы.</li>
          <li>Модель, обученная на RIAWELC (только метки патчей), работает в режиме <b>классификации патчей</b>: голова классификации на каждом тайле 256 px, необученный декодер не используется.</li>
          <li>Интерфейс декодирует снимок в браузере, поэтому нужен PNG или JPEG. TIFF работает при прямом обращении к API.</li>
          <li>Аутентификации нет. По умолчанию сервер слушает только localhost; для внешнего доступа нужен прокси с авторизацией.</li>
          <li>Пороги на <code>pe</code>/<code>mi</code> не откалиброваны. Выбирать их нужно на val, а не глядя на test.</li>
        </ul>
      </section>
    </>
  );
}
