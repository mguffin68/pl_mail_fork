<?php

declare(strict_types=1);

namespace DoctrineMigrations;

use Doctrine\DBAL\Schema\Schema;
use Doctrine\Migrations\AbstractMigration;

/**
 * Add the ai_backend column to ai_settings to support OpenAI-compatible APIs.
 */
final class Version20261010000307 extends AbstractMigration
{
    public function getDescription(): string
    {
        return 'Add ai_backend column to ai_settings for OpenAI-compatible API support';
    }

    public function up(Schema $schema): void
    {
        $this->addSql('ALTER TABLE ai_settings ADD ai_backend VARCHAR(16) NOT NULL DEFAULT \'ollama\'');
    }

    public function down(Schema $schema): void
    {
        $this->addSql('ALTER TABLE ai_settings DROP ai_backend');
    }
}
