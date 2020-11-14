const express = require('express')
const app = express()
const port = process.env.PORT || 3000

app.get('/', (req, res) => 
{
    res.send('Dummy server test!')
})

app.listen(port, () => 
{
    console.log(`Example app listening on ${port}`)
})