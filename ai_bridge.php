<?php
/**
 * SALINGO AI Bridge
 * ------------------
 * This file used to call Gemini/Groq/DeepSeek/OpenAI directly via cURL.
 * It now forwards requests to the SALINGO Translation Service (FastAPI,
 * LangChain + RAG translation memory) running on its own host.
 *
 * Function signatures are kept identical to what languageManagement.php
 * already expects, so no changes are needed there.
 */

// TODO: point this to wherever you deploy the Python service.
define('SALINGO_API_BASE', 'https://salingo-api.onrender.com');
define('SALINGO_API_TIMEOUT', 90);

/**
 * Trains (adds to the translation memory) a language using an uploaded
 * CSV file already saved on disk (e.g. "uploads/tagalog.csv").
 *
 * @param string $languageName
 * @param string $filePath   Local server path to the CSV file
 * @return array  ['success' => bool, 'count' => int, 'message' => string]
 */
function trainSalingoAI($languageName, $filePath) {
    if (!file_exists($filePath)) {
        return ['success' => false, 'count' => 0, 'message' => 'File not found: ' . $filePath];
    }

    $url = rtrim(SALINGO_API_BASE, '/') . '/train';

    $cfile = new CURLFile($filePath, 'text/csv', basename($filePath));
    $postFields = [
        'language' => $languageName,
        'file'     => $cfile,
    ];

    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $postFields);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, SALINGO_API_TIMEOUT);

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curlError = curl_error($ch);
    curl_close($ch);

    if ($response === false) {
        return ['success' => false, 'count' => 0, 'message' => 'cURL error: ' . $curlError];
    }

    $data = json_decode($response, true);

    if ($httpCode !== 200) {
        $msg = $data['detail'] ?? ('Training service returned HTTP ' . $httpCode);
        return ['success' => false, 'count' => 0, 'message' => $msg];
    }

    return [
        'success' => $data['success'] ?? false,
        'count'   => $data['count'] ?? 0,
        'message' => $data['message'] ?? '',
    ];
}


/**
 * Translates text using the trained translation memory + Gemini.
 *
 * @param string $text
 * @param string $languageName
 * @param string $direction   'to_english' or 'from_english'
 * @return array  ['success' => bool, 'translation' => string, 'message' => string]
 */
function translateText($text, $languageName, $direction = 'to_english') {
    $url = rtrim(SALINGO_API_BASE, '/') . '/translate';

    $payload = json_encode([
        'text'      => $text,
        'language'  => $languageName,
        'direction' => $direction,
    ]);

    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, SALINGO_API_TIMEOUT);

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curlError = curl_error($ch);
    curl_close($ch);

    if ($response === false) {
        return ['success' => false, 'translation' => '', 'message' => 'cURL error: ' . $curlError];
    }

    $data = json_decode($response, true);

    if ($httpCode !== 200) {
        $msg = $data['detail'] ?? ('Translation service returned HTTP ' . $httpCode);
        return ['success' => false, 'translation' => '', 'message' => $msg];
    }

    return [
        'success'     => true,
        'translation' => $data['translation'] ?? '',
        'message'     => '',
    ];
}


/**
 * Optional: list which languages currently have trained translation memory.
 */
function getTrainedLanguages() {
    $url = rtrim(SALINGO_API_BASE, '/') . '/languages';

    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, SALINGO_API_TIMEOUT);
    $response = curl_exec($ch);
    curl_close($ch);

    $data = json_decode($response, true);
    return $data['trained_languages'] ?? [];
}
