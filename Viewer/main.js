

// Function to render grid
const tile_size = 128;

var dom_selection_corners = [[null, null], [null, null]];
var selection_corner_tiles = [null, null];
var last_corner_selected = null;

var next_unique_tile_id = 0;

var tile_xy_grid = null;

var current_selection = [];

function get_tiles_in_xy_grid() {
    var tiles = Array.from(document.getElementsByClassName("tile"));
    // sort tiles by offsetTop
    var y_sorted_tiles = tiles;
    y_sorted_tiles.sort(
        function(a, b) {
            return a.offsetTop - b.offsetTop;
        }
    );
    var rows = [];
    var current_row = [];
    for (var i = 0; i < y_sorted_tiles.length; i++) {
        if (i == 0 || y_sorted_tiles[i].offsetTop == y_sorted_tiles[i-1].offsetTop) {
            current_row.push(y_sorted_tiles[i]);
        }
        else {
            rows.push(current_row);
            current_row = [y_sorted_tiles[i]];
        }
    }
    rows.push(current_row);
    // sort rows by offsetLeft
    var xy_sorted_tiles = [];
    for (var i = 0; i < rows.length; i++) {
        var row = rows[i];
        row.sort(
            function(a, b) {
                return a.offsetLeft - b.offsetLeft;
            }
        );
        xy_sorted_tiles.push(row);
    }
    return xy_sorted_tiles;
}

function dom_coord_by_tile(tile) {
    if (tile_xy_grid == null) {
        tile_xy_grid = get_tiles_in_xy_grid();
    }
    for (var i = 0; i < tile_xy_grid.length; i++) {
        for (var j = 0; j < tile_xy_grid[0].length; j++) {
            if (tile == tile_xy_grid[i][j]) {
                return [j, i];
            }
        }
    }
}

function dom_coord_by_id(id) {
    var rows = document.getElementsByClassName("row");
    for (var i = 0; i < rows.length; i++) {
        var tiles = rows[i].getElementsByClassName("tile");
        for (var j = 0; j < tiles.length; j++) {
            if (tiles[j].getAttribute("track_id") == id) {
                return [j, i];
            }
        }
    }
}

function changeCSS(typeAndClass, newRule, newValue) {
    var thisCSS=document.styleSheets[0]
    var ruleSearch=thisCSS.cssRules? thisCSS.cssRules: thisCSS.rules
    for (i=0; i<ruleSearch.length; i++)
    {
        if(ruleSearch[i].selectorText==typeAndClass)
        {
            var target=ruleSearch[i]
            break;
        }
    }
    target.style[newRule] = newValue;
}

function rowVisible(index) {
    if (index == -1) {
        index = document.getElementsByClassName('main_grid_row').length - 1;
    }
    var topMostVisible = document.getElementById('main_grid').scrollTop;
    var bottomMostVisible = topMostVisible + window.innerHeight;
    var row = document.getElementsByClassName('main_grid_row')[index];
    var row_top = row.offsetTop;
    var row_bottom = row_top + row.offsetHeight;
    var row_is_visible = (row_top < bottomMostVisible && row_bottom > topMostVisible);
    
    return row_is_visible;
  }

function colVisible(index) {
    if(index == -1){
        index = document.getElementsByClassName("main_grid_row")[0].getElementsByClassName("grid").length - 1;
    }
    var leftMostVisible = document.getElementById("main_grid").scrollLeft;
    var rightMostVisible = leftMostVisible + window.innerWidth;
    var subgrid = document.getElementsByClassName("main_grid_row")[0].getElementsByClassName("grid")[index];
    var subgrid_left = subgrid.offsetLeft;
    var subgrid_right = subgrid_left + subgrid.offsetWidth;
    var subgrid_is_visible = (subgrid_left < rightMostVisible && subgrid_right > leftMostVisible);
    return subgrid_is_visible;
}

function Subgrid() {
    // create grid element
    var grid = document.createElement('div');
    grid.className = 'grid';

    for (var i = 0; i < grid_ids.length; i++) {
        // Create div with class "row"
        var row = document.createElement('div');
        row.className = 'row';
        for (var j = 0; j < grid_ids[i].length; j++) {
            // Create div with background image track_images/{grid_ids[i][j]}.png
            var tilter = document.createElement('span');
            // tilter.className = "data-tilt";
            var tile = document.createElement('div');
            // Give classes "tile"
            tile.className = "tile";
            // Give attribute "track_id" with value grid_ids[i][j]
            tile.setAttribute("track_id", grid_ids[i][j]);
            tile.setAttribute("tile_id", next_unique_tile_id);
            next_unique_tile_id += 1;
            // Background image
            tile.style.backgroundImage = `url(track_images/${grid_ids[i][j]}.svg)`;
            // Append to row
            tilter.appendChild(tile);
            row.appendChild(tilter);
        }
        // Append to #grid
        grid.appendChild(row);
    }
    return grid;
}

function Main_Grid_Row(num_subgrids_x) {
    // Create div with class "main_grid_row"
    var main_grid_row = document.createElement('div');
    main_grid_row.className = 'main_grid_row';
    for (maingrid_col = 0; maingrid_col < num_subgrids_x; maingrid_col++) {
        subgrid_element = Subgrid();
        // Append to main_grid_row
        main_grid_row.appendChild(subgrid_element);
    }
    return main_grid_row;
}

function append_grid_row(num_cols=-1) {
    if (num_cols == -1) {
        num_cols = document.getElementsByClassName('main_grid_row')[0].getElementsByClassName('grid').length;
    }
    var row = Main_Grid_Row(num_cols);
    document.getElementById('main_grid').appendChild(row);
    tile_xy_grid = get_tiles_in_xy_grid();
}

function prepend_grid_row(num_cols=-1) {
    if (num_cols == -1) {
        num_cols = document.getElementsByClassName('main_grid_row')[0].getElementsByClassName('grid').length;
    }
    var row = Main_Grid_Row(num_cols);
    document.getElementById('main_grid').prepend(row);
    tile_xy_grid = get_tiles_in_xy_grid();
}

function append_grid_col() {
    var num_rows = document.getElementsByClassName('main_grid_row').length;
    for (var i = 0; i < num_rows; i++) {
        var subgrid_element = Subgrid();
        document.getElementsByClassName('main_grid_row')[i].appendChild(subgrid_element);
    }
    tile_xy_grid = get_tiles_in_xy_grid();
}

function prepend_grid_col() {
    var num_rows = document.getElementsByClassName('main_grid_row').length;
    for (var i = 0; i < num_rows; i++) {
        var subgrid_element = Subgrid();
        document.getElementsByClassName('main_grid_row')[i].prepend(subgrid_element);
    }
    tile_xy_grid = get_tiles_in_xy_grid();
}

function render_grid() {
    // add css rule for tile size
    changeCSS('.tile', 'width', `${tile_size}px`);
    changeCSS('.tile', 'height', `${tile_size}px`);

    // Find number of subgrids needed
    var window_width = window.innerWidth;
    var window_height = window.innerHeight;
    var grid_width = grid_ids[0].length * tile_size;
    var grid_height = grid_ids.length * tile_size;

    var num_subgrids_x = Math.ceil((window_width * 2) / grid_width);
    var num_subgrids_y = Math.ceil((window_height * 2) / grid_height);

    for (maingrid_row = 0; maingrid_row < num_subgrids_y; maingrid_row++) {
        append_grid_row(num_subgrids_x);
    }

    var info = {
        num_subgrids_x: num_subgrids_x,
        num_subgrids_y: num_subgrids_y,
        grid_width: grid_width,
        grid_height: grid_height
    }
    return info;
}

function modify_selection(){
    if (dom_selection_corners[0][0] == null || dom_selection_corners[0][1] == null || dom_selection_corners[1][0] == null || dom_selection_corners[1][1] == null) {
        return;
    }
    var selection_classes = ["selection_outline_left", "selection_outline_top", "selection_outline_right", "selection_outline_bottom", "in_selection", "within_x", "within_y"];
    var tiles = document.getElementsByClassName("tile");
    // Remove all selections
    current_selection = [];
    for (var i = 0; i < tiles.length; i++) {
        for (var j = 0; j < selection_classes.length; j++) {
            tiles[i].classList.remove(selection_classes[j]);
        }
    }
    // Find tiles strictly within selection, and add selection classes to all tiles with their track_id
    var minimum_x = Math.min(dom_selection_corners[0][0], dom_selection_corners[1][0]);
    var maximum_x = Math.max(dom_selection_corners[0][0], dom_selection_corners[1][0]);
    var minimum_y = Math.min(dom_selection_corners[0][1], dom_selection_corners[1][1]);
    var maximum_y = Math.max(dom_selection_corners[0][1], dom_selection_corners[1][1]);
    var strictly_selected_tiles = [];
    var strictly_left_tiles = [];
    var strictly_top_tiles = [];
    var strictly_right_tiles = [];
    var strictly_bottom_tiles = [];
    for (var i = 0; i < tiles.length; i++) {
        var coords = dom_coord_by_tile(tiles[i]);
        if (coords[0] >= minimum_x && coords[0] <= maximum_x && coords[1] >= minimum_y && coords[1] <= maximum_y) {
            strictly_selected_tiles.push(tiles[i]);
            if (coords[0] == minimum_x) {
                strictly_left_tiles.push(tiles[i]);
            }
            if (coords[0] == maximum_x) {
                strictly_right_tiles.push(tiles[i]);
            }
            if (coords[1] == minimum_y) {
                strictly_top_tiles.push(tiles[i]);
            }
            if (coords[1] == maximum_y) {
                strictly_bottom_tiles.push(tiles[i]);
            }
            current_selection.push(tiles[i].getAttribute("track_id"));
        }
    }
    for (var i = 0; i < strictly_left_tiles.length; i++) {
        var tiles_with_same_track_id = document.querySelectorAll(`[track_id="${strictly_left_tiles[i].getAttribute("track_id")}"]`);
        for (var j = 0; j < tiles_with_same_track_id.length; j++) {
            tiles_with_same_track_id[j].classList.add("selection_outline_left");
        }
        strictly_left_tiles[i].classList.add("selection_outline_left");
    }
    for (var i = 0; i < strictly_top_tiles.length; i++) {
        var tiles_with_same_track_id = document.querySelectorAll(`[track_id="${strictly_top_tiles[i].getAttribute("track_id")}"]`);
        for (var j = 0; j < tiles_with_same_track_id.length; j++) {
            tiles_with_same_track_id[j].classList.add("selection_outline_top");
        }
        strictly_top_tiles[i].classList.add("selection_outline_top");
    }
    for (var i = 0; i < strictly_right_tiles.length; i++) {
        var tiles_with_same_track_id = document.querySelectorAll(`[track_id="${strictly_right_tiles[i].getAttribute("track_id")}"]`);
        for (var j = 0; j < tiles_with_same_track_id.length; j++) {
            tiles_with_same_track_id[j].classList.add("selection_outline_right");
        }
        strictly_right_tiles[i].classList.add("selection_outline_right");
    }
    for (var i = 0; i < strictly_bottom_tiles.length; i++) {
        var tiles_with_same_track_id = document.querySelectorAll(`[track_id="${strictly_bottom_tiles[i].getAttribute("track_id")}"]`);
        for (var j = 0; j < tiles_with_same_track_id.length; j++) {
            tiles_with_same_track_id[j].classList.add("selection_outline_bottom");
        }
        strictly_bottom_tiles[i].classList.add("selection_outline_bottom");
    }
}

function select_tile(tile) {
    var dom_coords = dom_coord_by_tile(tile);
    if (last_corner_selected == null || last_corner_selected == 1) {
        last_corner_selected = 0;
        dom_selection_corners[0] = dom_coords;
        dom_selection_corners[1] = dom_coords;
    }
    else if (last_corner_selected == 0) {
        last_corner_selected = 1;
        dom_selection_corners[1] = dom_coords;
    }

    console.log(dom_selection_corners[0]);
    console.log(dom_selection_corners[1]);

    modify_selection();

    document.querySelector("#num_songs_selected").textContent = current_selection.length;
    console.log(current_selection);
}

function attach_tile_click_listener() {
    var tiles = document.getElementsByClassName("tile");
    for (var i = 0; i < tiles.length; i++) {
        tiles[i].onclick = function (e) {
            select_tile(e.target);
        }
    }
}

window.onload = function() { 
    var info = render_grid();
    var num_subgrids_x = info.num_subgrids_x;
    var num_subgrids_y = info.num_subgrids_y;
    var grid_width = info.grid_width;
    var grid_height = info.grid_height;
    var scrollX = num_subgrids_x * grid_width / 2 - window.innerWidth / 2;
    var scrollY = num_subgrids_y * grid_height / 2 - window.innerHeight / 2;
    modify_selection();
    attach_tile_click_listener();
    document.getElementById("main_grid").scrollTo(scrollX, scrollY);

    changeCSS(".tile", "display", "inline-block");
    tile_xy_grid = get_tiles_in_xy_grid();
}

function initialize_tilt() {
    // $(".tile").tilt({
    //     maxTilt: 40,
    //     perspective: 1400,
    //     easing: "cubic-bezier(.03,.98,.52,.99)",
    //     speed: 1200,
    //     glare: true,
    //     maxGlare: 0.2,
    //     scale: 1.1
    //   });
}

document.getElementById("main_grid").onscroll = function () {
    // console.log("Scrolling");
    // Check if last main_grid_row is in view
    if (rowVisible(-1)){
        append_grid_row();
        // console.log("Appended row");
    }
    // Check if first main_grid_row is in view
    if (rowVisible(0)){
        prepend_grid_row();
        var current_scroll_left = document.getElementById("main_grid").scrollLeft;
        var current_scroll_top = document.getElementById("main_grid").scrollTop;
        var grid_height = document.getElementsByClassName("main_grid_row")[0].offsetHeight;
        document.getElementById("main_grid").scrollTo(current_scroll_left, current_scroll_top + grid_height);
        // console.log("Prepended row");
    }
    // Check if last column is in view
    if (colVisible(-1)){
        append_grid_col();
        // console.log("Appended col");
    }
    // Check if first column is in view
    if (colVisible(0)){
        prepend_grid_col();
        var current_scroll_left = document.getElementById("main_grid").scrollLeft;
        var current_scroll_top = document.getElementById("main_grid").scrollTop;
        var grid_width = document.getElementsByClassName("main_grid_row")[0].getElementsByClassName("grid")[0].offsetWidth;
        document.getElementById("main_grid").scrollTo(current_scroll_left + grid_width, current_scroll_top);
        // console.log("Prepended col");
    }
    modify_selection();
    attach_tile_click_listener();
}

function getTileAtPos(x, y) {
    var grid_height = document.getElementsByClassName("main_grid_row")[0].offsetHeight;
    var grid_width = document.getElementsByClassName("main_grid_row")[0].getElementsByClassName("grid")[0].offsetWidth;

    var main_grid_y = Math.floor(y / grid_height);
    var main_grid_x = Math.floor(x / grid_width);

    var main_grid_row = document.getElementsByClassName("main_grid_row")[main_grid_y];
    var subgrid = main_grid_row.getElementsByClassName("grid")[main_grid_x];

    var offset_y_in_grid = y % grid_height;
    var offset_x_in_grid = x % grid_width;

    var tile_y = Math.floor(offset_y_in_grid / tile_size);
    var tile_x = Math.floor(offset_x_in_grid / tile_size);

    var tile = subgrid.getElementsByClassName("row")[tile_y].getElementsByClassName("tile")[tile_x];
    return tile;
}

function assignTileToPlayer(tile, position) {
    var tiles = document.getElementsByClassName("tile");
    for (var i = 0; i < tiles.length; i++) {
        tiles[i].classList.remove(position + "_tile");
    }
    tile.classList.add(position + "_tile");
    var player = document.getElementById(position + "_audio");
    var track_id = tile.getAttribute("track_id");
    
    // Check if some other player is playing the same track
    var players = document.getElementsByTagName("audio");
    var resumeTime = 0;
    for (var i = 0; i < players.length; i++) {
        if (players[i].getAttribute("src") == `track_clips/${track_id}.mp3`) {
            resumeTime = players[i].currentTime;
        }
    }

    // Check if this player is playing the same track
    if (player.getAttribute("src") == `track_clips/${track_id}.mp3`) {
        return;
    }
    else {
        player.pause();
        player.setAttribute("src", `track_clips/${track_id}.mp3`);
        player.load();
        player.currentTime = resumeTime;
        player.play();
    }
}

function bilinearInterpolate(x, y) {

    // Get absolute value of x and y
    x = Math.abs(x); 
    y = Math.abs(y);

    if (x > 0 && y > 0) {
        var cUpperLeft = 0;
        var cUpperCenter = 0;
        var cLeft = 0;
        var cCenter = Math.sqrt((1-x)**2 + (1-y)**2);
        var cRight = x;
        var cUpperRight = 0;
        var cLowerLeft = 0;
        var cLowerCenter = y;
        var cLowerRight = Math.sqrt(x**2 + y**2);
    }

    else if (x > 0) {
        y = -y;
        var cUpperLeft = 0;
        var cUpperCenter = y;
        var cLeft = 0;
        var cCenter = Math.sqrt((1-x)**2 + (1-y)**2);
        var cRight = 1 - x;
        var cUpperRight = Math.sqrt(x**2 + y**2);
        var cLowerLeft = 0;
        var cLowerCenter = 0;
        var cLowerRight = 0;
    }
    else if (y > 0) {
        x = -x;
        var cUpperLeft = 0;
        var cUpperCenter = 0;
        var cLeft = x;
        var cCenter = Math.sqrt((1-x)**2 + (1-y)**2);
        var cRight = 0;
        var cUpperRight = 0;
        var cLowerLeft = Math.sqrt(x**2 + y**2);
        var cLowerCenter = y;
        var cLowerRight = 0;
    }
    else if (x < 0 && y < 0) {
        x = -x;
        y = -y;
        var cUpperLeft = Math.sqrt(x**2 + y**2);
        var cUpperCenter = y;
        var cLeft = x;
        var cCenter = Math.sqrt((1-x)**2 + (1-y)**2);
        var cRight = 0;
        var cUpperRight = 0;
        var cLowerLeft = 0;
        var cLowerCenter = 0;
        var cLowerRight = 0;
    }
    else {
        var cUpperLeft = 0;
        var cUpperCenter = 0;
        var cLeft = 0;
        var cCenter = 1;
        var cRight = 0;
        var cUpperRight = 0;
        var cLowerLeft = 0;
        var cLowerCenter = 0;
        var cLowerRight = 0;
    }
  
    // Return coefficients as object
    return {
      cUpperLeft, cUpperCenter,
      cLowerLeft, cLowerCenter,
      cLeft, cCenter, 
      cRight, cUpperRight, cLowerRight
    };
  }

document.getElementById("main_grid").onmousemove = function (e) {
    var scrollTop = document.getElementById("main_grid").scrollTop;
    var scrollLeft = document.getElementById("main_grid").scrollLeft;

    var center_tile = getTileAtPos(e.clientX + scrollLeft, e.clientY + scrollTop);

    assignTileToPlayer(center_tile, "center");

    // console.log(dom_coord_by_tile(center_tile));

    // document.getElementById("tooltip").textContent = center_tile.getAttribute("track_id");
    // document.getElementById("tooltip").style.top = e.clientY + 10 + "px";
    // document.getElementById("tooltip").style.left = e.clientX + 10 + "px";
}



function close_modal() {
    initialize_tilt();
    document.getElementById("splash_modal").style.display = "none";
}

document.getElementById("autoplay_modal_no").onclick = function () {
    close_modal();
}

document.getElementById("autoplay_modal_yes").onclick = function () {
    close_modal();
    var audios = document.getElementsByTagName("audio");
    for (var i = 0; i < audios.length; i++) {
        audios[i].muted = false;
    }
}

document.getElementById("close_tips_button").onclick = function () {
    document.getElementById("tips").classList.remove("open");
    document.getElementById("tips").classList.add("closed");
}

document.getElementById("open_tips_button").onclick = function () {
    document.getElementById("tips").classList.remove("closed");
    document.getElementById("tips").classList.add("open");
}

document.getElementById("create_playlist_button").onclick = function () {
    var song_urls = [];
    for (var i = 0; i < current_selection.length; i++) {
        song_urls.push(`https://open.spotify.com/track/${current_selection[i]}`);
    }
    document.getElementById("playlist_modal").getElementsByTagName("textarea")[0].value = song_urls.join("\n");
    document.getElementById("playlist_modal").classList.add("open");
    document.getElementById("playlist_modal").classList.remove("closed");
}

document.getElementById("close_playlist_modal_button").onclick = function () {
    document.getElementById("playlist_modal").classList.add("closed");
    document.getElementById("playlist_modal").classList.remove("open");
}

document.getElementById("playlist_modal_copy_button").onclick = function () {
    document.getElementById("playlist_modal_textarea").select();
    document.execCommand("copy");
}