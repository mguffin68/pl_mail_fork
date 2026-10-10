<?php

declare(strict_types=1);

namespace App\Service\Ai;

use App\Domain\DTO\Ai\AiChatResult;
use App\Domain\DTO\Ai\AiEmbedResult;
use App\Domain\DTO\Ai\AiProbe;
use App\Domain\DTO\Ai\AiWarmUp;
use App\Domain\DTO\Ai\LoadedModel;

/**
 * Abstract interface for AI backend clients.
 *
 * Allows plMail to work with either Ollama or any OpenAI-compatible API
 * without changing the application logic that calls these methods.
 */
interface AiClientInterface
{
    /**
     * What models are currently loaded in memory.
     *
     * @return list<LoadedModel>
     */
    public function ps(string $baseUrl, ?float $timeout = null): array;

    /**
     * Is the host reachable and what models does it list?
     */
    public function probe(string $baseUrl, ?float $timeout = null): AiProbe;

    /**
     * Generate an embedding vector for a piece of text.
     */
    public function embed(string $baseUrl, string $model, string $text, ?string $keepAlive = null): AiEmbedResult;

    /**
     * A single non-streamed chat completion.
     *
     * @param list<array{role: string, content: string}> $messages
     */
    public function chat(string $baseUrl, string $model, array $messages, ?float $temperature = null, ?string $keepAlive = null): AiChatResult;

    /**
     * A streamed chat completion.
     *
     * @param list<array{role: string, content: string}> $messages
     *
     * @return \Generator<int, string, void, AiChatResult>
     */
    public function chatStream(string $baseUrl, string $model, array $messages, ?float $temperature = null, ?string $keepAlive = null, ?int $numCtx = null, ?float $timeout = null): \Generator;

    /**
     * Load a model into memory without generating anything.
     */
    public function preload(string $baseUrl, string $model, ?string $keepAlive = null): AiWarmUp;
}
