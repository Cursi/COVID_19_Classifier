const express = require('express')
const app = express()
const port = process.env.PORT || 3000

const fileUpload = require('express-fileupload');
app.use(fileUpload());

const pyCmd = require("node-cmd");

app.get('/', (req, res) => 
{
    res.sendFile(__dirname + "/index.html");
})

app.post("/upload", (req, res) =>
{
    console.log(req.files);

    if(req.files)
    {
        let file = req.files.dataset;
        let filename = file.name;

        file.mv("./upload/" + filename, (err) =>
        {
            if(err)
            {
                console.log(err);
                res.send(err);
            }
            else
            {
                pyCmd.run(`python dummy.py ${__dirname}/upload/${filename}`).stdout.on('data', function(data) 
                {
                    console.log(data.toString());
                    res.write(data);
                    res.end();
                });
            }
        });
    }
});

app.listen(port, () => 
{
    console.log(`Example app listening on ${port}`);
})