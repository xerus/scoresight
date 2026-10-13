# Using the ScoreSight HTTP Server

Get your scoreboard information in the browser for a simple integration into many streaming software, like OBS.

This feature is only available on ScoreSight Pro.

To get started, set up your scoreboard with the information you want to extract. See the Setup Process Guide for instructions.

Enable the HTTP Server from the bottom-right section of ScoreSight: "Start HTTP Server"

You can then open any browser and enter "http://localhost:18099/scoresight" as the URL. A simple scoreboard will appear. You may now use this URL in your streaming software.

All the scoreboard information will be delivered as JSON in http://localhost:18099/json . This can be used as API in any other software or even HTML pages.

## JSON response formats

`GET /json` returns a list of fields. Each entry contains `name`, `text`, `state`,
and `rect` (`x`, `y`, `width`, `height`). It includes rejected readings and empty
fields so clients can inspect their state. Text values remain strings: a score
of zero is `"0"`, a confirmed blank is `""`, and a missing raw reading can be
JSON `null`. A client should check `state` before publishing a reading.

`GET /json?pivot` returns a map from field names to text. It includes `Success`
and `SameNoChange` readings, plus `Empty` readings when **Skip Empty Values** is
off. `FailedFilter` readings and skipped blanks are omitted. For example:

```json
{"Home Score": "0", "Away Score": "", "Time": "01:07"}
```

The presence of `pivot` enables this format, even with `?pivot=false`. With no
results, the responses are `[]` and `{}` respectively. Duplicate names use the
last accepted value in pivot output; use distinct box names to retain all fields.
Each update replaces the previous batch, so clients must handle omitted fields
explicitly rather than assuming that an earlier score is still current.

The JSON endpoint accepts GET requests and browser CORS/preflight requests.
It does not accept POST updates. Its regression tests cover both response formats,
UTF-8 text and JSON escaping, output-state filtering, fresh batch replacement,
and a real HTTP connection on an automatically allocated local port. CI runs
these tests on Windows and Linux.

If you want to create a more customized HTML scoreboard you are able to do so. Create a ".html" file anywhere on your disk and use the template example below to fetch data automatically from ScoreSight into the browser.

Here is a simple HTML example that uses the JSON output:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scoreboard</title>
<style>
  body { font-family: Arial, sans-serif; text-align: center; }
  .scoreboard { display: flex; justify-content: normal; align-items: center; margin-top: 20px; }
  .team { width: 25%; }
  .info { background: #f0f0f0; padding: 20px; border-radius: 10px; }
  .team-name { font-size: 24px; margin-bottom: 10px; }
  .score { font-size: 48px; margin-bottom: 10px; }
  .timer { font-size: 78px; margin: 0 20px; }
  .fouls { font-size: 24px; }
</style>
</head>
<body>
<script>
  function updateScoreboard() {
    fetch('http://localhost:18099/json')
      .then(response => response.json())
      .then(data => {
      const homeScore = data.find(item => item.name === 'Home Score').text;
      const awayScore = data.find(item => item.name === 'Away Score').text;
      const time = data.find(item => item.name === 'Time').text;
      const homeFouls = data.find(item => item.name === 'Home Fouls').text;
      const awayFouls = data.find(item => item.name === 'Away Fouls').text;

      document.querySelector('.team.home .score').textContent = homeScore;
      document.querySelector('.team.away .score').textContent = awayScore;
      document.querySelector('.timer').textContent = time;
      document.querySelector('.team.home .fouls').textContent = `Fouls: ${homeFouls}`;
      document.querySelector('.team.away .fouls').textContent = `Fouls: ${awayFouls}`;
      })
      .catch(error => console.error('Error:', error));
  }

  setInterval(updateScoreboard, 100);
</script>

<div class="scoreboard">
  <div class="team home">
    <div class="info">
      <div class="team-name">Home Team</div>
      <div class="score">0</div>
      <div class="fouls">Fouls: 0</div>
    </div>
  </div>
  <div class="timer">00:00</div>
  <div class="team away">
    <div class="info">
      <div class="team-name">Away Team</div>
      <div class="score">0</div>
      <div class="fouls">Fouls: 0</div>
    </div>
  </div>
</div>

</body>
</html>
```
