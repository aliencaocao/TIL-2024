# NLP

## Input

Text transcription of a turret voice command.

Example transcription: `"Target is red helicopter, heading is zero one zero, tool to deploy is surface-to-air missiles."`

## Output

JSON object containing three keys:

1. "target", for the target identified
2. "heading", for the heading of the target
3. "tool", for the tool to be deployed to neutralize the target

Example JSON:

```json
{
  "target": "red helicopter",
  "heading": "010",
  "tool": "surface-to-air missiles"
}
```
