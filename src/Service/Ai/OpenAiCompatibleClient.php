<?php

declare(strict_types=1);

namespace App\Service\Ai;

use App\Domain\DTO\Ai\AiCallTiming;
use App\Domain\DTO\Ai\AiChatResult;
use App\Domain\DTO\Ai\AiEmbedResult;
use App\Domain\DTO\Ai\AiProbe;
use App\Domain\DTO\Ai\AiWarmUp;
use App\Domain\DTO\Ai\OllamaModel;
use Psr\Log\LoggerInterface;
use Symfony\Contracts\HttpClient\Exception\DecodingExceptionInterface;
use Symfony\Contracts\HttpClient\Exception\ExceptionInterface as HttpClientException;
use Symfony\Contracts\HttpClient\Exception\TimeoutExceptionInterface;
use Symfony\Contracts\HttpClient\Exception\TransportExceptionInterface;
use Symfony\Contracts\HttpClient\HttpClientInterface;
use Throwable;

/**
 * Talks to a model host that speaks the OpenAI /v1 dialect.
 *
 * LM Studio, llama.cpp's server, vLLM, LocalAI, text-generation-webui — and
 * Ollama itself, which serves /v1 beside its native /api. Same network policy
 * as OllamaClient, for the same reasons: the address is typed by an
 * administrator and the private address IS the feature. See OllamaClient.
 *
 *   GET  /v1/models            — probe()
 *   POST /v1/embeddings        — embed()
 *   POST /v1/chat/completions  — chat(), chatStream(), preload()
 *
 * THE HOST MAY BE GIVEN WITH OR WITHOUT /v1
 * ─────────────────────────────────────────
 * Every one of these servers documents its address as `http://host:port/v1`,
 * so that is what people paste. Both forms work; see url().
 *
 * WHAT THE DIALECT CANNOT SAY
 * ───────────────────────────
 * There is no /api/ps, so ps() answers "nothing resident" and the composer
 * assumes a cold load — the safe guess. There is no keep_alive either; how
 * long a model stays loaded is the host's own setting. And there is no
 * preload, so preload() asks for a one-token completion instead, which loads
 * the model as a side effect on every server listed above.
 */
final readonly class OpenAiCompatibleClient implements AiClientInterface
{
    /** The same closed set as OllamaClient, because both feed AiCallRecorder. */
    public const string ERROR_TIMEOUT      = OllamaClient::ERROR_TIMEOUT;
    public const string ERROR_UNREACHABLE  = OllamaClient::ERROR_UNREACHABLE;
    public const string ERROR_HTTP_404     = OllamaClient::ERROR_HTTP_404;
    public const string ERROR_HTTP_STATUS  = OllamaClient::ERROR_HTTP_STATUS;
    public const string ERROR_BAD_RESPONSE = OllamaClient::ERROR_BAD_RESPONSE;
    public const string ERROR_UNEXPECTED   = OllamaClient::ERROR_UNEXPECTED;

    /** See OllamaClient — every one of these is an IDLE timeout. */
    public const float GENERATE_TIMEOUT = 120.0;

    private const float HEARTBEAT_SECONDS = 10.0;

    private const float EMBED_TIMEOUT = 60.0;

    private const float PROBE_TIMEOUT = 5.0;

    private const float PRELOAD_TIMEOUT = 240.0;

    public function __construct(
        private HttpClientInterface $http,
        private LoggerInterface     $logger,
        private ?string             $token = null,
    ) {
    }

    /**
     * The same client, sending this token as a bearer.
     *
     * LM Studio, vLLM and LocalAI can all be told to require a key, so the
     * token that is optional for Ollama is routinely needed here.
     */
    public function withToken(?string $token): self
    {
        $token = null === $token ? null : trim($token);

        return new self($this->http, $this->logger, '' === $token ? null : $token);
    }

    /** No /api/ps in this dialect: nothing is known to be resident. */
    public function ps(string $baseUrl, ?float $timeout = null): array
    {
        return [];
    }

    public function probe(string $baseUrl, ?float $timeout = null): AiProbe
    {
        try {
            $response = $this->http->request('GET', $this->url($baseUrl, '/models'), $this->options([
                'timeout' => $timeout ?? self::PROBE_TIMEOUT,
            ]));

            $status = $response->getStatusCode();

            if (200 !== $status) {
                return AiProbe::unreachable('status', ['status' => $status]);
            }

            /** @var array{data?: list<array{id?: mixed, owned_by?: mixed}>} $payload */
            $payload = $response->toArray(false);

            $models = [];

            foreach ($payload['data'] ?? [] as $entry) {
                $id = $entry['id'] ?? null;

                if (false === is_string($id) || '' === $id) {
                    continue;
                }

                $models[] = new OllamaModel(
                    name:   $id,
                    family: is_string($entry['owned_by'] ?? null) ? $entry['owned_by'] : null,
                );
            }

            return AiProbe::reachable($models);
        } catch (HttpClientException $exception) {
            return AiProbe::unreachable('unreachable', ['error' => $exception->getMessage()]);
        } catch (Throwable $exception) {
            $this->logger->error('OpenAiCompatibleClient: probe failed unexpectedly', [
                'error'     => $exception->getMessage(),
                'exception' => $exception,
            ]);

            return AiProbe::unreachable('unreachable', ['error' => $exception->getMessage()]);
        }
    }

    /** $keepAlive is ignored: the dialect has no such field. */
    public function embed(string $baseUrl, string $model, string $text, ?string $keepAlive = null): AiEmbedResult
    {
        try {
            $response = $this->http->request('POST', $this->url($baseUrl, '/embeddings'), $this->options([
                'json'    => ['model' => $model, 'input' => $this->utf8($text, $model)],
                'timeout' => self::EMBED_TIMEOUT,
            ]));

            $status = $response->getStatusCode();

            if (200 !== $status) {
                $this->logger->warning('OpenAiCompatibleClient: embed refused', [
                    'model'  => $model,
                    'status' => $status,
                    'error'  => self::refusal($response->getContent(false)),
                ]);

                return AiEmbedResult::failed(404 === $status ? self::ERROR_HTTP_404 : self::ERROR_HTTP_STATUS);
            }

            /** @var array{data?: list<array{embedding?: mixed}>, usage?: mixed} $body */
            $body = $response->toArray(false);

            $vector = $body['data'][0]['embedding'] ?? null;

            if (false === is_array($vector) || [] === $vector) {
                return AiEmbedResult::failed(self::ERROR_BAD_RESPONSE);
            }

            return AiEmbedResult::ok(array_map(floatval(...), array_values($vector)), self::timing($body));
        } catch (TimeoutExceptionInterface) {
            return AiEmbedResult::failed(self::ERROR_TIMEOUT);
        } catch (TransportExceptionInterface) {
            return AiEmbedResult::failed(self::ERROR_UNREACHABLE);
        } catch (DecodingExceptionInterface) {
            return AiEmbedResult::failed(self::ERROR_BAD_RESPONSE);
        } catch (Throwable $exception) {
            $this->logger->error('OpenAiCompatibleClient: embed failed unexpectedly', [
                'model'     => $model,
                'error'     => $exception->getMessage(),
                'exception' => $exception,
            ]);

            return AiEmbedResult::failed(self::ERROR_UNEXPECTED);
        }
    }

    /**
     * @param list<array{role: string, content: string}> $messages
     */
    public function chat(string $baseUrl, string $model, array $messages, ?float $temperature = null, ?string $keepAlive = null): AiChatResult
    {
        $payload = [
            'model'    => $model,
            'messages' => $this->cleanMessages($messages, $model),
            'stream'   => false,
        ];

        if (null !== $temperature) {
            $payload['temperature'] = $temperature;
        }

        try {
            $response = $this->http->request('POST', $this->url($baseUrl, '/chat/completions'), $this->options([
                'json'    => $payload,
                'timeout' => self::GENERATE_TIMEOUT,
            ]));

            $status = $response->getStatusCode();

            if (200 !== $status) {
                $this->logger->warning('OpenAiCompatibleClient: chat refused', [
                    'model'  => $model,
                    'status' => $status,
                    'error'  => self::refusal($response->getContent(false)),
                ]);

                return AiChatResult::failed(404 === $status ? self::ERROR_HTTP_404 : self::ERROR_HTTP_STATUS);
            }

            /** @var array{choices?: list<array{message?: array{content?: mixed}}>, usage?: mixed} $body */
            $body = $response->toArray(false);

            $content = $body['choices'][0]['message']['content'] ?? null;

            if (false === is_string($content) || '' === trim($content)) {
                return AiChatResult::failed(self::ERROR_BAD_RESPONSE);
            }

            return AiChatResult::ok($content, self::timing($body));
        } catch (TimeoutExceptionInterface $exception) {
            $this->logger->warning('OpenAiCompatibleClient: chat timed out', [
                'model'     => $model,
                'error'     => $exception->getMessage(),
                'exception' => $exception,
            ]);

            return AiChatResult::failed(self::ERROR_TIMEOUT);
        } catch (TransportExceptionInterface $exception) {
            $this->logger->warning('OpenAiCompatibleClient: chat could not reach the host', [
                'model'     => $model,
                'error'     => $exception->getMessage(),
                'exception' => $exception,
            ]);

            return AiChatResult::failed(self::ERROR_UNREACHABLE);
        } catch (DecodingExceptionInterface $exception) {
            $this->logger->warning('OpenAiCompatibleClient: chat answered with something that is not JSON', [
                'model'     => $model,
                'error'     => $exception->getMessage(),
                'exception' => $exception,
            ]);

            return AiChatResult::failed(self::ERROR_BAD_RESPONSE);
        } catch (Throwable $exception) {
            $this->logger->error('OpenAiCompatibleClient: chat failed unexpectedly', [
                'model'     => $model,
                'error'     => $exception->getMessage(),
                'exception' => $exception,
            ]);

            return AiChatResult::failed(self::ERROR_UNEXPECTED);
        }
    }

    /**
     * The same completion, token by token, as server-sent events.
     *
     * Same contract as OllamaClient::chatStream(), including the empty-string
     * heartbeat, and the same two-loop shape for the same reasons — read the
     * comments there before simplifying this. In short: stream() ENDS on a
     * timeout chunk, so waiting longer means calling it again; and the status
     * is read on the first chunk, because getStatusCode() blocks until the
     * headers arrive and a host loading a model sends none for a long time.
     *
     * $keepAlive and $numCtx are ignored: the dialect has neither, and the
     * context window is the host's own load-time setting.
     *
     * @param list<array{role: string, content: string}> $messages
     *
     * @return \Generator<int, string, void, AiChatResult>
     */
    public function chatStream(string $baseUrl, string $model, array $messages, ?float $temperature = null, ?string $keepAlive = null, ?int $numCtx = null, ?float $timeout = null): \Generator
    {
        $payload = [
            'model'    => $model,
            'messages' => $this->cleanMessages($messages, $model),
            'stream'   => true,
        ];

        if (null !== $temperature) {
            $payload['temperature'] = $temperature;
        }

        $whole   = '';
        $timing  = AiCallTiming::none();
        $budget  = $timeout ?? self::GENERATE_TIMEOUT;
        $started = microtime(true);

        try {
            $response = $this->http->request('POST', $this->url($baseUrl, '/chat/completions'), $this->options([
                'json'    => $payload,
                'timeout' => self::HEARTBEAT_SECONDS,
            ]));

            $buffer = '';

            while (true) {
                $timedOut = false;

                foreach ($this->http->stream($response) as $chunk) {
                    if (true === $chunk->isTimeout()) {
                        $timedOut = true;

                        break;
                    }

                    if (true === $chunk->isFirst()) {
                        $status = $response->getStatusCode();

                        if (200 !== $status) {
                            $this->logger->error('OpenAiCompatibleClient: streamed chat refused', [
                                'model'  => $model,
                                'status' => $status,
                                'error'  => self::refusal($response->getContent(false)),
                            ]);

                            return AiChatResult::failed(
                                404 === $status ? self::ERROR_HTTP_404 : self::ERROR_HTTP_STATUS,
                            );
                        }

                        continue;
                    }

                    // The last chunk carries no content, and a host that does
                    // not end its final event with a newline leaves it in the
                    // buffer — so one more newline flushes it through below.
                    $last    = $chunk->isLast();
                    $buffer .= $last ? "\n" : $chunk->getContent();

                    while (false !== ($break = strpos($buffer, "\n"))) {
                        $line   = trim(substr($buffer, 0, $break));
                        $buffer = substr($buffer, $break + 1);

                        // SSE: `data:` with or without a space, and comment
                        // lines (`: keep-alive`) that some servers send while
                        // the model loads.
                        if (false === str_starts_with($line, 'data:')) {
                            continue;
                        }

                        $data = ltrim(substr($line, 5));

                        if ('[DONE]' === $data) {
                            break 3;
                        }

                        $object = json_decode($data, true);

                        if (false === is_array($object)) {
                            continue;
                        }

                        $delta = $object['choices'][0]['delta']['content'] ?? null;

                        if (is_string($delta) && '' !== $delta) {
                            $whole .= $delta;

                            yield $delta;
                        }

                        if (is_array($object['usage'] ?? null)) {
                            $timing = self::timing($object);
                        }
                    }

                    if ($last) {
                        break 2;
                    }
                }

                if (false === $timedOut) {
                    break;
                }

                $waited = microtime(true) - $started;

                if ($waited > $budget) {
                    $this->logger->error('OpenAiCompatibleClient: gave up waiting for a streamed answer', [
                        'model'          => $model,
                        'waited_seconds' => round($waited),
                        'budget_seconds' => round($budget),
                        'chars_so_far'   => mb_strlen($whole),
                    ]);

                    $response->cancel();

                    return AiChatResult::failed(self::ERROR_TIMEOUT, $timing);
                }

                yield '';
            }
        } catch (TimeoutExceptionInterface $exception) {
            $this->logger->error('OpenAiCompatibleClient: a streamed chat timed out', [
                'model'          => $model,
                'waited_seconds' => round(microtime(true) - $started),
                'budget_seconds' => round($budget),
                'error'          => $exception->getMessage(),
                'exception'      => $exception,
            ]);

            return AiChatResult::failed(self::ERROR_TIMEOUT, $timing);
        } catch (TransportExceptionInterface $exception) {
            $this->logger->error('OpenAiCompatibleClient: a streamed chat lost its connection', [
                'model'          => $model,
                'waited_seconds' => round(microtime(true) - $started),
                'chars_so_far'   => mb_strlen($whole),
                'error'          => $exception->getMessage(),
                'exception'      => $exception,
            ]);

            return AiChatResult::failed(self::ERROR_UNREACHABLE, $timing);
        } catch (Throwable $exception) {
            $this->logger->error('OpenAiCompatibleClient: streamed chat failed unexpectedly', [
                'model'     => $model,
                'error'     => $exception->getMessage(),
                'exception' => $exception,
            ]);

            return AiChatResult::failed(self::ERROR_UNEXPECTED, $timing);
        }

        if ('' === trim($whole)) {
            return AiChatResult::failed(self::ERROR_BAD_RESPONSE, $timing);
        }

        return AiChatResult::ok($whole, $timing);
    }

    /**
     * Load the model by asking it for one token.
     *
     * There is no preload in this dialect, but every server that speaks it
     * loads the model on the first completion — LM Studio's JIT loading,
     * Ollama's /v1, llama-swap — so a one-token answer is a warm-up that
     * costs one token. Reasons match OllamaClient::preload(), so the admin
     * page's messages apply unchanged.
     */
    public function preload(string $baseUrl, string $model, ?string $keepAlive = null): AiWarmUp
    {
        $started = microtime(true);

        try {
            $response = $this->http->request('POST', $this->url($baseUrl, '/chat/completions'), $this->options([
                'json' => [
                    'model'      => $model,
                    'messages'   => [['role' => 'user', 'content' => 'Hi']],
                    'max_tokens' => 1,
                    'stream'     => false,
                ],
                'timeout' => self::PRELOAD_TIMEOUT,
            ]));

            $status = $response->getStatusCode();

            if (200 !== $status) {
                $error = self::refusal($response->getContent(false));

                // OpenAI's own wording for a missing model is 404, but several
                // local servers answer 400 with the model's name in the body.
                if (404 === $status || (400 === $status && str_contains(strtolower($error), 'model'))) {
                    return AiWarmUp::failed('model_missing', ['model' => $model, 'error' => $error], self::elapsed($started));
                }

                return AiWarmUp::failed('status', ['status' => $status, 'error' => $error], self::elapsed($started));
            }

            return AiWarmUp::loaded(self::elapsed($started));
        } catch (TimeoutExceptionInterface) {
            return AiWarmUp::failed('timeout', [], self::elapsed($started));
        } catch (TransportExceptionInterface $exception) {
            return AiWarmUp::failed('unreachable', ['error' => $exception->getMessage()], self::elapsed($started));
        } catch (Throwable $exception) {
            $this->logger->error('OpenAiCompatibleClient: warm-up failed unexpectedly', [
                'model'     => $model,
                'error'     => $exception->getMessage(),
                'exception' => $exception,
            ]);

            return AiWarmUp::failed('unexpected', ['error' => $exception->getMessage()], self::elapsed($started));
        }
    }

    /**
     * Request options with the bearer token, when there is one.
     *
     * @param array<string, mixed> $options
     *
     * @return array<string, mixed>
     */
    private function options(array $options): array
    {
        if (null !== $this->token) {
            $options['auth_bearer'] = $this->token;
        }

        return $options;
    }

    /**
     * Token counts only: this dialect reports no durations.
     *
     * @param array<array-key, mixed> $body
     */
    private static function timing(array $body): AiCallTiming
    {
        $usage = $body['usage'] ?? null;

        if (false === is_array($usage)) {
            return AiCallTiming::none();
        }

        return new AiCallTiming(
            promptTokens: is_int($usage['prompt_tokens'] ?? null) ? $usage['prompt_tokens'] : null,
            evalTokens:   is_int($usage['completion_tokens'] ?? null) ? $usage['completion_tokens'] : null,
        );
    }

    private static function elapsed(float $started): int
    {
        return (int) round((microtime(true) - $started) * 1000);
    }

    /** A refusal body, clipped — a proxy may answer with a whole HTML page. */
    private static function refusal(string $body): string
    {
        $body = trim($body);

        return mb_strlen($body) > 300 ? mb_substr($body, 0, 300) . '…' : $body;
    }

    /**
     * @param list<array{role: string, content: string}> $messages
     *
     * @return list<array{role: string, content: string}>
     */
    private function cleanMessages(array $messages, string $model): array
    {
        return array_map(
            fn (array $message): array => [
                'role'    => $message['role'],
                'content' => $this->utf8($message['content'], $model),
            ],
            $messages,
        );
    }

    /** See OllamaClient::utf8() — one bad byte fails json_encode for the whole request. */
    private function utf8(string $text, string $model): string
    {
        if (mb_check_encoding($text, 'UTF-8')) {
            return $text;
        }

        $scrubbed = mb_scrub($text, 'UTF-8');

        $this->logger->warning('OpenAiCompatibleClient: repaired malformed UTF-8 before sending', [
            'model' => $model,
            'bytes' => strlen($text) - strlen($scrubbed),
        ]);

        return $scrubbed;
    }

    /**
     * The endpoint under /v1, whether or not the administrator typed /v1.
     */
    private function url(string $baseUrl, string $path): string
    {
        $base = rtrim(trim($baseUrl), '/');

        if (false === str_ends_with(strtolower($base), '/v1')) {
            $base .= '/v1';
        }

        return $base . $path;
    }
}
