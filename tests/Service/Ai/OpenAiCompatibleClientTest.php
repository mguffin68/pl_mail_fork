<?php

declare(strict_types=1);

namespace App\Tests\Service\Ai;

use App\Service\Ai\OpenAiCompatibleClient;
use PHPUnit\Framework\TestCase;
use Psr\Log\NullLogger;
use Symfony\Component\HttpClient\MockHttpClient;
use Symfony\Component\HttpClient\Response\MockResponse;

/**
 * The /v1 dialect: LM Studio, llama.cpp, vLLM, and Ollama's own /v1.
 *
 * The stream is server-sent events rather than Ollama's one-object-per-line,
 * so the split-chunk test from OllamaStreamTest is repeated here against the
 * other framing — it is the same bug in a different costume.
 */
final class OpenAiCompatibleClientTest extends TestCase
{
    private const array HI = [['role' => 'user', 'content' => 'hi']];

    /** @var list<array{method: string, url: string, options: array<string, mixed>}> */
    private array $requests = [];

    public function testTheHostWorksWithOrWithoutV1(): void
    {
        $client = $this->client([$this->models(), $this->models(), $this->models()]);

        $client->probe('http://h:1234');
        $client->probe('http://h:1234/v1');
        $client->probe('http://h:1234/v1/');

        self::assertSame(
            ['http://h:1234/v1/models', 'http://h:1234/v1/models', 'http://h:1234/v1/models'],
            array_column($this->requests, 'url'),
        );
    }

    public function testProbeListsTheModelsTheHostServes(): void
    {
        $probe = $this->client([$this->models()])->probe('http://h:1234');

        self::assertTrue($probe->reachable);
        self::assertTrue($probe->hasModel('qwen2.5-7b-instruct'));
        self::assertFalse($probe->hasModel('llama3.1'));
    }

    public function testTheTokenIsSentAsABearer(): void
    {
        $this->client([$this->models()])->withToken('  sk-local  ')->probe('http://h:1234');

        self::assertContains('Authorization: Bearer sk-local', $this->requests[0]['options']['normalized_headers']['authorization'] ?? []);
    }

    public function testNoTokenSendsNoAuthorisation(): void
    {
        $this->client([$this->models()])->withToken('')->probe('http://h:1234');

        self::assertArrayNotHasKey('authorization', $this->requests[0]['options']['normalized_headers'] ?? []);
    }

    public function testEmbedReturnsTheFirstVector(): void
    {
        $result = $this->client([new MockResponse((string) json_encode([
            'data'  => [['embedding' => [0.25, -1, 0.5]]],
            'usage' => ['prompt_tokens' => 3],
        ]))])->embed('http://h:1234', 'e', 'text');

        self::assertTrue($result->succeeded);
        self::assertSame([0.25, -1.0, 0.5], $result->vector);
        self::assertSame(3, $result->timing->promptTokens);
    }

    public function testChatReturnsTheMessage(): void
    {
        $result = $this->client([new MockResponse((string) json_encode([
            'choices' => [['message' => ['role' => 'assistant', 'content' => 'Dear Paul']]],
            'usage'   => ['prompt_tokens' => 12, 'completion_tokens' => 2],
        ]))])->chat('http://h:1234', 'm', self::HI);

        self::assertTrue($result->succeeded);
        self::assertSame('Dear Paul', $result->content);
        self::assertSame(2, $result->timing->evalTokens);
    }

    public function testStreamedTokensArriveAndUsageIsKept(): void
    {
        $body = implode("\n\n", [
            ': loading model',
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"Dear "}}]}',
            'data:{"choices":[{"delta":{"content":"Paul"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}',
            'data: [DONE]',
        ]) . "\n\n";

        $tokens = $this->client([new MockResponse($body)])->chatStream('http://h:1234', 'm', self::HI);

        self::assertSame(['Dear ', 'Paul'], iterator_to_array($tokens, false));

        $result = $tokens->getReturn();

        self::assertTrue($result->succeeded);
        self::assertSame('Dear Paul', $result->content);
        self::assertSame(2, $result->timing->evalTokens);
    }

    /** An event split across two chunks must survive the join. */
    public function testAChunkThatSplitsAnEventIsNotLost(): void
    {
        $tokens = $this->client([new MockResponse([
            'data: {"choices":[{"delta":{"content":"Hel',
            'lo"}}]}' . "\n\n" . 'data: {"choices":[{"delta":{"con',
            'tent":" there"}}]}',
        ])])->chatStream('http://h:1234', 'm', self::HI);

        // No trailing newline and no [DONE]: the last event is still read.
        self::assertSame(['Hello', ' there'], iterator_to_array($tokens, false));
        self::assertSame('Hello there', $tokens->getReturn()->content);
    }

    public function testAMissingModelIsSaidSoDistinctly(): void
    {
        $tokens = $this->client([new MockResponse('{"error":"model not found"}', ['http_code' => 404])])
            ->chatStream('http://h:1234', 'm', self::HI);

        iterator_to_array($tokens, false);

        self::assertSame(OpenAiCompatibleClient::ERROR_HTTP_404, $tokens->getReturn()->errorKind);
    }

    public function testWarmUpAsksForOneToken(): void
    {
        $warm = $this->client([new MockResponse('{"choices":[{"message":{"content":"H"}}]}')])
            ->preload('http://h:1234/v1', 'm');

        self::assertTrue($warm->loaded);
        self::assertSame('http://h:1234/v1/chat/completions', $this->requests[0]['url']);
        self::assertStringContainsString('"max_tokens":1', (string) $this->requests[0]['options']['body']);
    }

    public function testWarmUpOfAMissingModelSaysWhichOne(): void
    {
        $warm = $this->client([new MockResponse('{"error":"model \"m\" not found"}', ['http_code' => 400])])
            ->preload('http://h:1234', 'm');

        self::assertFalse($warm->loaded);
        self::assertSame('model_missing', $warm->reason);
    }

    public function testNothingIsEverResident(): void
    {
        self::assertSame([], $this->client([])->ps('http://h:1234'));
        self::assertSame([], $this->requests);
    }

    /**
     * @param list<MockResponse> $responses
     */
    private function client(array $responses): OpenAiCompatibleClient
    {
        $http = new MockHttpClient(function (string $method, string $url, array $options) use (&$responses): MockResponse {
            $this->requests[] = ['method' => $method, 'url' => $url, 'options' => $options];

            return array_shift($responses) ?? new MockResponse('', ['http_code' => 500]);
        });

        return new OpenAiCompatibleClient($http, new NullLogger());
    }

    private function models(): MockResponse
    {
        return new MockResponse((string) json_encode([
            'object' => 'list',
            'data'   => [['id' => 'qwen2.5-7b-instruct', 'object' => 'model', 'owned_by' => 'organization_owner']],
        ]));
    }
}
