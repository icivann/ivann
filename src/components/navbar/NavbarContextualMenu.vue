<template>
    <div id="contextual-menu">
      <div v-for="(editor, index) in editors" :key="index">
        <VerticalMenuButton
          :label="editor.name"
          :onClick="() => switchEditor({ editorType, index})"
          :isSelected="editorType === currEditorType && index === currEditorIndex">
        </VerticalMenuButton>
      </div>
      <VerticalMenuButton
        :label="'+'"
        :onClick="this.createNewEditor"
        :isSelected="false"
      />
    </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import VerticalMenuButton from '@/components/buttons/VerticalMenuButton.vue';
import { mapGetters, mapMutations } from 'vuex';
import EditorType from '@/EditorType';
import { Getter, Mutation } from 'vuex-class';
import { EditorModel } from '@/store/editors/types';
import { uniqueTextInput } from '@/inputs/prompt';
import { saveEditor, SaveWithNames } from '@/file/EditorAsJson';

@Component({
  components: { VerticalMenuButton },
  computed: mapGetters([
    'currEditorType',
    'currEditorIndex',
  ]),
  methods: mapMutations(['switchEditor']),
})
export default class NavbarContextualMenu extends Vue {
  @Prop({ required: true }) readonly editors!: EditorModel[];
  @Prop({ required: true }) readonly editorType!: EditorType;
  @Getter('editorNames') editorNames!: Set<string>;
  @Getter('saveWithNames') saveWithNames!: SaveWithNames;
  @Getter('currEditorModel') currEditorModel!: EditorModel;
  @Mutation('newEditor') newEditor!: (arg0: { editorType: EditorType; name: string}) => void;

  private createNewEditor(): void {
    const name: string | null = uniqueTextInput(this.editorNames,
      'Please enter a unique name for the editor');
    if (name !== null) {
      this.newEditor({ editorType: this.editorType, name });
      // Auto-Saving this new Editor
      this.$cookies.set('unsaved-project', this.saveWithNames);
      this.$cookies.set(`unsaved-editor-${name}`, saveEditor(this.currEditorModel));
    }
  }
}

</script>

<style scoped>
  #contextual-menu {
    border: 1px solid var(--grey);
  }
</style>
