package in.dljava.gnotebook.book;

import java.util.prefs.Preferences;

import in.dljava.gnotebook.book.storage.BookManager;
import in.dljava.gnotebook.controls.Pages;
import in.dljava.gnotebook.controls.StatusBar;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SeparatorMenuItem;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class BookView extends Application {

	private Stage stage;

	private BookManager bookManager = new BookManager();

	@Override
	public void start(Stage stage) {

		this.stage = stage;

		StatusBar sb = new StatusBar();
		sb.textProperty().bind(bookManager.getBookMessageProperty());

		Pages pages = new Pages(bookManager);

		BorderPane vb = new BorderPane();
		vb.setTop(this.getMenuBar());
		vb.setCenter(pages);
		vb.setBottom(sb);

		Scene scene = new Scene(vb);
		scene.getStylesheets().add(getClass().getResource("/gbook.css").toExternalForm());

		stage.setTitle("Groovy Notebook");
		stage.setScene(scene);

		Preferences userPrefs = Preferences.userNodeForPackage(getClass());

		stage.setX(userPrefs.getDouble("stage.x", 100));
		stage.setY(userPrefs.getDouble("stage.y", 100));
		stage.setWidth(userPrefs.getDouble("stage.width", 500));
		stage.setHeight(userPrefs.getDouble("stage.height", 500));

		stage.show();
	}

	private MenuBar getMenuBar() {

		Menu file = new Menu("File");
		MenuItem m4 = new MenuItem("New book");
		MenuItem m1 = new MenuItem("Open book");
		MenuItem m2 = new MenuItem("Save book");
		MenuItem m3 = new MenuItem("Close");
		m3.setOnAction(e -> bookManager.closeGroovyBook());

		file.getItems().addAll(m4, new SeparatorMenuItem(), m1, m2, new SeparatorMenuItem(), m3);

		Menu blocks = new Menu("Block");
		m1 = new MenuItem("Add code block above");
		m2 = new MenuItem("Add code block below");

		MenuItem m5 = new MenuItem("Add markdown block above");
		MenuItem m6 = new MenuItem("Add markdown block below");

		m3 = new MenuItem("Delete current block");

		m4 = new MenuItem("Toggle code / markdown block");

		blocks.getItems().addAll(m1, m2, new SeparatorMenuItem(), m5, m6, new SeparatorMenuItem(), m3,
				new SeparatorMenuItem(), m4);

		Menu run = new Menu("Run");
		m1 = new MenuItem("Run all blocks");
		m2 = new MenuItem("Run current block");
		m3 = new MenuItem("Run blocks below");

		run.getItems().addAll(m1, new SeparatorMenuItem(), m2, m3);

		MenuBar mb = new MenuBar();

		mb.getMenus().add(file);
		mb.getMenus().add(blocks);
		mb.getMenus().add(run);

		return mb;
	}

	@Override
	public void stop() {

		Preferences userPrefs = Preferences.userNodeForPackage(getClass());

		userPrefs.putDouble("stage.x", stage.getX());
		userPrefs.putDouble("stage.y", stage.getY());
		userPrefs.putDouble("stage.width", stage.getWidth());
		userPrefs.putDouble("stage.height", stage.getHeight());
	}

	public static void startBook(String... args) {
		launch(args);
	}
}
