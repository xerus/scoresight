# vMix Integration

ScoreSight can interact directly with vMix through an API, providing a smooth and fast connection to vMix scoreboard Titles. Here is how to get it up and running.

Enable the Web Controller in **vMix Settings → Web Controller**. ScoreSight uses the HTTP API on the same address as the Web Controller, normally port **8088**. Confirm that the Web Controller opens in a browser from the ScoreSight machine. See the [vMix HTTP API documentation](https://www.vmix.com/help29/DeveloperAPI.html).

![alt text](image-8.png)

In a vMix input add a Title that fits your needs, for example this scoreboard (which is available in vMix):

![alt text](image-9.png)

Right click and open the Title Editor:

![alt text](image-10.png)

Note the names of the various texts in the Title, these would be controlled by ScoreSight.

In ScoreSight setup your scoreboard with detection boxes:

![alt text](image-11.png)

Go on the vMix tab and update or inspect the connection information like host and port (8088 is the default HTTP port), and choose the vMix Input according to your vMix setup. Next, set the mapping between ScoreSight detections and the vMix Title texts according to the information in the Title Editor (seen above).

![alt text](image-12.png)

Make sure the mappings are exactly as they appear in the Title Editor, including case. GT title text fields need the `.Text` suffix, for example `HomeScore.Text`. XAML titles use their text element name. Input can be the number, full input title, or its GUID. Leave a mapping cell empty to disable that field. These parameters follow the [vMix HTTP API](https://www.vmix.com/help29/DeveloperAPI.html).

Once the mapping is complete, click **Start vMix**. New accepted readings update the mapped title fields. Enable **Send Same Values** to also refresh unchanged readings. Confirmed blanks clear a field unless **Skip Empty Values** is enabled; rejected readings preserve the previous field value.

![alt text](image-13.png)

ScoreSight sends `SetText` requests in a background worker with connection and read timeouts. If vMix is slower than OCR, pending updates retain the latest value for each field. Failed requests are logged and retried on the next OCR update, including unchanged readings. Stopping or changing connection/mapping settings discards queued requests; a request already sent cannot be recalled.

Verification covers API parameters, Unicode escaping, result states, mapping changes, connection failures, background delivery, and a local HTTP server. Live vMix on Windows still needs a manual check: set two mapped fields including a GT `.Text` field, verify changing numbers and blanks, stop/start output, and temporarily disconnect vMix to confirm recovery. An actual vMix installation was unavailable for the automated tests.
