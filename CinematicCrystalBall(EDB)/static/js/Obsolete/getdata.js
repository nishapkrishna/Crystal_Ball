
fetch('/get_all_data')
  .then(function(response) {
    return response.json();
  })
  .then(function(myJson) {
    //console.log(JSON.stringify(myJson));
    html = '<table class="table table-bordered table-hover table-condensed"><thead><tr><th>State</th><th>Year</th><th>Avg Listing Price</th><th>Active Listing Count</th></tr></thead><tbody>';
    for(var i = 0; i < myJson.length; i++) {
        var obj = myJson[i];
        tableRow = '<tr><td>'+obj['State']+ '</td><td>' + obj['Year'] + '</td><td>' + obj['Avg Listing price'] + '</td><td>' + obj['Active Listing Count'] + '</td></tr>' ;1
        console.log (tableRow);
        html += tableRow ;
    }
    html += '</tbody></table>';
    $("#divResults").append(html);
  });