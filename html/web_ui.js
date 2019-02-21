var WebUI = function (host, uid) {
  var websocket_type = mpl.get_websocket_type();
  var keepalive = null;
  var all_figures = {};

  function savefig(figure, format) {
    window.open('/'+figure.id+'/download.'+format, '_blank');
  }

  function add_figure(fignum) {
    var ws = new websocket_type('ws://'+host+'/'+uid+'/'+fignum+'/ws');
    var elt = $('#fig'+fignum);
    all_figures[fignum] = new mpl.figure(fignum, ws, savefig, elt);
  }
  function on_form_submit(evt) {
    evt.preventDefault();
    var form = $(this);
    var btn = form.find('input:last');
    btn.val("Running...").prop('disabled', true);
    var res_div = form.next('.results');
    // close old websockets, if needed
    res_div.find('.figure').each(function(i,elt){
      var fig = all_figures[elt.id.slice(3)];
      if (fig){ fig.ws.close(); }
    });
    var post_data = new FormData(this);
      post_data.append('uid', uid);
    // do $.load() manually, to use the FormData object
    $.ajax({
      url: '/',
      data: post_data,
      processData: false,
      contentType: false,
      dataType: 'html',
      type: 'POST',
    }).done(function(responseText) {
      // add the html
      res_div.html(responseText);
      // add any new figures
      $('.figure', res_div).each(function(){
        add_figure($(this).attr('id').slice(3));
      });
      // scroll to put the results on the page
      var det = form.parent('details');
      det[0].scrollIntoView(false);
      // open next section and enable its button
      var nxt = det.next('details');
      nxt.attr('open', true);
      nxt.find('form input:last').prop('disabled', false);
    }).fail(function(jqXHR) {
      // show the error message
      res_div.html('<span class="err_msg">' + jqXHR.responseText + '</span>');
    }).always(function() {
      // re-enable the run button
      btn.val("Run").prop('disabled', false);
    });
  }

  // return the onready function for this module
  return function() {
    // open keep-alive websocket
    keepalive = new websocket_type('ws://'+host+'/'+uid+'/0/ws');
    keepalive.onclose = function(){
      // close all figures by replacing their canvases with static images
      $('.figure').each(function(){
        var img = new Image();
        img.src = $(this).find('.mpl-canvas')[0].toDataURL('image/png');
        $(this).html(img);
      });
      // throw away the dict of mpl.figure objects
      all_figures = {};
    };
    // intercept form submission
      $('form').submit(on_form_submit);
      
  };
};
